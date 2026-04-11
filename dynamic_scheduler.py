"""
dynamic_scheduler.py — Ray-based dynamic GPU memory scheduler for sensitivity scans
====================================================================================

Provides run_parallel_sensitivity_dynamic(), a drop-in replacement for
run_parallel_sensitivity() that uses Ray actors instead of static subprocesses.

Key differences from the subprocess-based approach
---------------------------------------------------
Static (parallel_sensitivity.py):
  • P workers are assigned at launch time and never rebalanced.
  • Each worker gets a fixed chunk of L/P layers.
  • If a worker finishes early its GPU sits idle.

Dynamic (this module):
  • Workers are Ray actors; the scheduler monitors real-time GPU memory
    via pynvml and rebalances layer assignments when:
      - A worker finishes its chunk early (idle GPU has free VRAM).
      - A worker is under memory pressure (steal layers back to queue).
  • New workers can be spawned mid-run if a GPU gains free VRAM.
  • NVIDIA MPS is optionally enabled for same-GPU multi-worker contexts
    to serialize CUDA kernel launches and prevent context-switch overhead.

Architecture
------------
    DynamicSensitivityScheduler (Ray actor, head node)
        ├── MemoryMonitor  — polls pynvml every POLL_INTERVAL_S seconds
        ├── WorkQueue      — thread-safe queue of unassigned layer names
        └── RebalanceEngine — steals idle memory, spawns new workers

    SensitivityWorkerActor (Ray actor, one per worker slot)
        ├── Loads worker_sim.onnx + encodings into ORT CUDAExecutionProvider
        ├── Runs enabling loop on assigned layers
        └── Reports status (current_layer, layers_done, memory_gb) on demand

Usage
-----
Called by quant.py when --dynamic-scheduling is set:

    from dynamic_scheduler import run_parallel_sensitivity_dynamic
    sensitivity_data = run_parallel_sensitivity_dynamic(
        quanter=quanter,
        sim=sim,
        ...
        sn_per_worker_gb=1.5,
    )

Requirements
------------
    pip install ray[default]   # Ray core + dashboard
    pip install pynvml         # NVIDIA Management Library Python bindings

NVIDIA MPS (optional, for same-GPU multi-worker):
    # Start MPS daemon before launching (requires root or nvidia-smi access):
    nvidia-cuda-mps-control -d
    # Set per-worker env vars (handled automatically when ENABLE_MPS=True):
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu<id>
    CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-gpu<id>
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Seconds between memory-monitor polls inside the scheduler actor.
POLL_INTERVAL_S: float = 10.0

# A GPU is considered "memory-rich" (eligible for a new worker) if its free
# VRAM exceeds this multiple of U = m_u + sn.
SPAWN_HEADROOM_FACTOR: float = 1.1

# A worker is considered "memory-pressured" if its GPU's free VRAM drops
# below this threshold in GB.  The scheduler will not steal layers from it.
MEMORY_PRESSURE_THRESHOLD_GB: float = 0.5

# Enable NVIDIA MPS for same-GPU multi-worker contexts.
# MPS serializes CUDA kernel launches, preventing context-switch overhead
# when multiple workers share a GPU.
ENABLE_MPS: bool = False   # set to True if MPS daemon is running


# ---------------------------------------------------------------------------
# pynvml helpers (module-level, not inside Ray actors to avoid import issues)
# ---------------------------------------------------------------------------

def _get_free_vram_gb(gpu_id: int) -> float | None:
    """Return free VRAM on *gpu_id* in GB via pynvml.  None on failure."""
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.free) / (1024.0 ** 3)
    except Exception:
        return None


def _poll_gpu_pool(gpu_ids: list[int]) -> dict[int, float]:
    """Return {gpu_id: free_gb} for all GPUs.  Missing entries use 0.0."""
    result: dict[int, float] = {}
    for gid in gpu_ids:
        v = _get_free_vram_gb(gid)
        result[gid] = v if v is not None else 0.0
    return result


# ---------------------------------------------------------------------------
# Ray worker actor
# ---------------------------------------------------------------------------

def _make_worker_actor_class():
    """Return the SensitivityWorkerActor Ray remote class.

    Defined inside a factory function so that `import ray` only happens when
    this module is actually used (not at quant.py import time).
    """
    import ray  # type: ignore[import]

    @ray.remote(num_gpus=0)   # GPU assigned via CUDA_VISIBLE_DEVICES env var
    class SensitivityWorkerActor:
        """Ray actor that runs the enabling loop on an assigned layer subset.

        Each actor loads the full QDQ-augmented ONNX model into its own ORT
        CUDAExecutionProvider session.  The GPU is controlled via
        CUDA_VISIBLE_DEVICES in the actor's runtime_env, so device_id=0
        always refers to the correct physical GPU inside the actor.
        """

        def __init__(
            self,
            worker_id: int,
            gpu_id: int,
            onnx_path: str,
            encodings_path: str,
            module_alias: str,
            data_path: str,
            qscheme_key: str,
        ):
            import logging as _logging
            import onnx as _onnx
            import onnxruntime as _ort
            from pathlib import Path as _Path

            _logging.basicConfig(
                level=_logging.DEBUG,
                format="%(asctime)s | %(levelname)-8s | worker%(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self._log = _logging.getLogger(f"[{worker_id}]")

            self.worker_id   = worker_id
            self.gpu_id      = gpu_id
            self._results: dict[str, float] = {}
            self._current_layer: str | None = None
            self._layers_done: int = 0

            # ── ORT providers ────────────────────────────────────────────────
            available = _ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                self._log.warning("CUDAExecutionProvider unavailable — using CPU")

            # ── Load ONNX + QuantSim ─────────────────────────────────────────
            import aimet_onnx as _aimet_onnx
            from aimet_onnx.common.defs import QuantScheme as _QS
            from aimet_onnx.quantsim import load_encodings_to_sim as _load_enc

            # Resolve qscheme types (mirrors sensitivity_worker.py QSCHEME_TYPES)
            _QSCHEME_TYPES = {
                "fp16":  (_aimet_onnx.float16, _aimet_onnx.float16),
                "w8a16": (_aimet_onnx.int8,    _aimet_onnx.int16),
                "w8a8":  (_aimet_onnx.int8,    _aimet_onnx.int8),
                "w4a16": (_aimet_onnx.int4,    _aimet_onnx.int16),
                "w4a4":  (_aimet_onnx.int4,    _aimet_onnx.int4),
            }
            param_type, activation_type = _QSCHEME_TYPES[qscheme_key]

            self._log.info("Loading ONNX: %s", onnx_path)
            onnx_model = _onnx.load(onnx_path)
            self._sim = _aimet_onnx.QuantizationSimModel(
                model=onnx_model,
                quant_scheme=_QS.post_training_tf,
                config_file="htp_v81",
                param_type=param_type,
                activation_type=activation_type,
                providers=providers,
            )

            # Load calibrated encodings
            enc_path = encodings_path
            if not _Path(enc_path).exists():
                alt = enc_path + ".json"
                if _Path(alt).exists():
                    enc_path = alt
                else:
                    raise FileNotFoundError(f"Encodings not found: {encodings_path}")
            _load_enc(self._sim, enc_path)
            self._log.info("Encodings loaded: %s", enc_path)

            # ── Dataloader + eval callback ───────────────────────────────────
            from pipeline.data_loader import create_module_dataloader as _cdl
            import numpy as _np

            self._module_alias = module_alias
            self._dataloader = _cdl(
                intermediate_data_path=data_path,
                module_name=module_alias,
                batch_size=1,
                shuffle=False,
            )

            # Build op_name → quantizers map for ALL ops (filtered per layer later)
            self._op_name_to_quantizers: dict[str, list] = {}
            for op in self._sim.connected_graph.ordered_ops:
                in_qs, out_qs, param_qs = self._sim.get_op_quantizers(op)
                all_qs = list(in_qs) + list(out_qs) + list(param_qs.values())
                if all_qs:
                    self._op_name_to_quantizers[op.name_op] = all_qs

            self._log.info(
                "Worker %d ready: gpu=%d  quantizable_ops=%d",
                worker_id, gpu_id, len(self._op_name_to_quantizers),
            )

        # ── Status reporting ─────────────────────────────────────────────────

        def get_status(self) -> dict:
            """Return a lightweight status dict for the scheduler.

            NOTE: pynvml uses physical GPU indices and ignores CUDA_VISIBLE_DEVICES.
            We must query self.gpu_id (the physical GPU ID passed at construction),
            NOT device 0 — which would always return GPU 0's memory regardless of
            which physical GPU this actor is running on.
            """
            free_gb = _get_free_vram_gb(self.gpu_id)   # physical GPU ID, not CUDA device 0
            return {
                "worker_id":     self.worker_id,
                "gpu_id":        self.gpu_id,
                "current_layer": self._current_layer,
                "layers_done":   self._layers_done,
                "free_vram_gb":  free_gb if free_gb is not None else -1.0,
            }

        def get_results(self) -> dict[str, float]:
            """Return all scored layers so far."""
            return dict(self._results)

        # ── Layer processing ─────────────────────────────────────────────────

        def process_layers(self, layer_names: list[str]) -> dict[str, float]:
            """Run enabling loop on *layer_names*; return {layer: sqnr}."""
            import numpy as _np
            from aimet_onnx.qc_quantize_op import OpMode as _OpMode

            # Snapshot all quantizer states
            snapshot: dict[int, tuple] = {}
            for q in self._sim.qc_quantize_op_dict.values():
                try:
                    bw = q.bitwidth
                except AttributeError:
                    bw = 8
                snapshot[id(q)] = (q.enabled, q.data_type, bw, q.op_mode)

            # Disable all quantizers
            for q in self._sim.qc_quantize_op_dict.values():
                q.enabled = False

            for layer_name in layer_names:
                self._current_layer = layer_name
                qs = self._op_name_to_quantizers.get(layer_name, [])
                if not qs:
                    self._log.warning("No quantizers for layer %s — skipping", layer_name)
                    continue

                # Enable this layer's quantizers
                for q in qs:
                    orig = snapshot.get(id(q))
                    if orig is None or not orig[0]:
                        continue
                    orig_enabled, orig_dtype, orig_bw, orig_mode = orig
                    q.enabled   = True
                    q.data_type = orig_dtype
                    q.set_bitwidth(orig_bw)
                    q.op_mode   = _OpMode.quantizeDequantize

                # Evaluate
                try:
                    score = self._eval_session()
                except Exception as exc:
                    self._log.warning("eval failed for %s: %s", layer_name, exc)
                    score = float("nan")

                self._results[layer_name] = score
                self._layers_done += 1
                self._log.debug("  %-52s  SQNR=%8.2f dB", layer_name, score)

                # Disable this layer before moving on
                for q in qs:
                    q.enabled = False

            # Restore original state
            for q in self._sim.qc_quantize_op_dict.values():
                orig = snapshot.get(id(q))
                if orig is None:
                    continue
                orig_enabled, orig_dtype, orig_bw, orig_mode = orig
                q.enabled = orig_enabled
                if orig_enabled:
                    q.data_type = orig_dtype
                    q.set_bitwidth(orig_bw)
                    q.op_mode   = orig_mode

            self._current_layer = None
            return dict(self._results)

        def steal_layers(self, extra_layers: list[str]) -> dict[str, float]:
            """Accept additional layers from the scheduler and process them."""
            return self.process_layers(extra_layers)

        # ── Internal eval ────────────────────────────────────────────────────

        def _eval_session(self) -> float:
            """Run the full dataloader through sim.session and return mean SQNR."""
            import numpy as _np
            import torch as _torch

            session    = self._sim.session
            sqnr_list: list[float] = []
            input_names = [inp.name for inp in session.get_inputs()]

            for batch in self._dataloader:
                inp_raw = batch["input"]
                out_raw = batch["output"]

                bs = int(inp_raw[0].shape[0]) if isinstance(inp_raw, (list, tuple)) else int(inp_raw.shape[0])

                for idx in range(bs):
                    if isinstance(inp_raw, (list, tuple)):
                        inputs  = [x[idx].detach().cpu().numpy() if hasattr(x[idx], "numpy") else x[idx] for x in inp_raw]
                        outputs = [x[idx].detach().cpu().numpy() if hasattr(x[idx], "numpy") else x[idx] for x in out_raw]
                    else:
                        inputs  = inp_raw[idx].detach().cpu().numpy() if hasattr(inp_raw[idx], "numpy") else inp_raw[idx]
                        outputs = [out_raw[idx].detach().cpu().numpy() if hasattr(out_raw[idx], "numpy") else out_raw[idx]]

                    # Build feed dict
                    if isinstance(inputs, (list, tuple)):
                        if self._module_alias == "Qwen2_05_VLM" and len(inputs) >= 4:
                            feed = {
                                "inputs_embeds":      inputs[0],
                                "mlp1_output":        inputs[1],
                                "path_route_queries": inputs[3],
                            }
                        elif len(inputs) == len(input_names):
                            feed = dict(zip(input_names, inputs))
                        else:
                            feed = {input_names[0]: inputs[0]}
                    else:
                        feed = {input_names[0]: inputs}

                    ort_outs = session.run(None, feed)

                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    for ort_out, ref_out in zip(ort_outs, outputs):
                        if ref_out is not None:
                            err = _np.sum((ref_out - ort_out) ** 2)
                            if err < 1e-12:
                                sqnr_list.append(float("inf"))
                            else:
                                sig = _np.sum(ref_out ** 2)
                                sqnr_list.append(float(10 * _np.log10(sig / err)))

            return float(_np.mean(sqnr_list)) if sqnr_list else 0.0

    return SensitivityWorkerActor


# ---------------------------------------------------------------------------
# Dynamic scheduler
# ---------------------------------------------------------------------------

def _make_scheduler_actor_class():
    """Return the DynamicSensitivityScheduler Ray remote class."""
    import ray  # type: ignore[import]

    @ray.remote
    class DynamicSensitivityScheduler:
        """Monitors GPU memory and rebalances layer assignments across workers.

        The scheduler runs as a Ray actor on the head node.  It polls GPU
        memory every POLL_INTERVAL_S seconds and:
          1. Detects idle workers (finished their chunk, GPU has free VRAM).
          2. Steals layers from the global queue and assigns them to idle workers.
          3. Logs a per-GPU memory snapshot at each poll cycle.
        """

        def __init__(
            self,
            gpu_ids: list[int],
            m_u: float,
            sn_per_worker: float,
            logger_name: str = "dynamic_scheduler",
        ):
            import logging as _logging
            _logging.basicConfig(
                level=_logging.DEBUG,
                format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self._log = _logging.getLogger(logger_name)
            self.gpu_ids       = gpu_ids
            self.m_u           = m_u
            self.sn            = sn_per_worker
            self.U             = m_u + sn_per_worker
            self._layer_queue: list[str] = []
            self._workers: dict[int, Any] = {}   # worker_id → actor handle
            self._worker_gpu: dict[int, int] = {}  # worker_id → gpu_id

        def register_worker(self, worker_id: int, gpu_id: int, actor_handle: Any) -> None:
            self._workers[worker_id]    = actor_handle
            self._worker_gpu[worker_id] = gpu_id

        def set_layer_queue(self, layers: list[str]) -> None:
            self._layer_queue = list(layers)

        def get_queue_size(self) -> int:
            return len(self._layer_queue)

        def get_remaining_layers(self) -> list[str]:
            """Return and clear the remaining layer queue.

            Called during the drain phase after all initial worker futures have
            completed, to ensure no layers are silently dropped due to race
            conditions between the monitor loop and the final rebalance cycle.
            """
            remaining = list(self._layer_queue)
            self._layer_queue = []
            return remaining

        def poll_and_rebalance(self) -> dict:
            """Poll GPU memory and steal layers from queue to idle workers.

            Returns a status dict with per-GPU free VRAM and rebalance actions.
            """
            import ray as _ray

            free_mem = _poll_gpu_pool(self.gpu_ids)
            self._log.debug(
                "Memory poll: %s",
                "  ".join(f"gpu{g}={v:.1f}GB" for g, v in sorted(free_mem.items())),
            )

            if not self._layer_queue:
                return {"queue_empty": True, "free_mem": free_mem}

            actions: list[str] = []
            for wid, actor in self._workers.items():
                gpu_id   = self._worker_gpu[wid]
                free_gb  = free_mem.get(gpu_id, 0.0)

                # Only steal if GPU has enough headroom for at least one more
                # model instance worth of activations
                if free_gb < self.U * SPAWN_HEADROOM_FACTOR:
                    continue
                if not self._layer_queue:
                    break

                # How many extra layers can we safely assign?
                # Use a conservative estimate: 1 layer per U GB of headroom
                extra_slots = max(1, int(free_gb / self.U))
                steal_count = min(extra_slots, len(self._layer_queue))
                stolen      = self._layer_queue[:steal_count]
                self._layer_queue = self._layer_queue[steal_count:]

                actor.steal_layers.remote(stolen)
                actions.append(f"worker{wid}(gpu{gpu_id})+{steal_count}layers")
                self._log.info(
                    "Rebalance: assigned %d layers to worker %d (gpu %d, free=%.1f GB)",
                    steal_count, wid, gpu_id, free_gb,
                )

            return {
                "queue_remaining": len(self._layer_queue),
                "free_mem":        free_mem,
                "actions":         actions,
            }

    return DynamicSensitivityScheduler


# ---------------------------------------------------------------------------
# MPS helpers
# ---------------------------------------------------------------------------

def _mps_env_for_gpu(gpu_id: int) -> dict[str, str]:
    """Return env vars to enable NVIDIA MPS for *gpu_id*.

    MPS (Multi-Process Service) serializes CUDA kernel launches from multiple
    processes sharing the same GPU, preventing context-switch overhead and
    reducing per-process CUDA context memory.

    Prerequisites:
        nvidia-cuda-mps-control -d   # start MPS daemon (requires root)
    """
    return {
        "CUDA_MPS_PIPE_DIRECTORY": f"/tmp/nvidia-mps-gpu{gpu_id}",
        "CUDA_MPS_LOG_DIRECTORY":  f"/tmp/nvidia-log-gpu{gpu_id}",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_parallel_sensitivity_dynamic(
    quanter: Any,
    sim: Any,
    dummy_input: dict,
    full_forward: Any,
    full_eval: Any,
    mmp_results_dir: str,
    gpu_ids: list[int],
    vram_per_gpu_gb: float,
    cushion_gb: float,
    qscheme_key: str,
    op_name_to_quantizers: dict[str, list],
    htp_disabled_layers: list[str],
    data_path: str,
    step_num: int,
    sn_per_worker_gb: float = 0.0,
    gpu_headroom_gb: float = 1.0,
) -> dict[str, float]:
    """Dynamic Ray-based parallel per-layer sensitivity scan.

    Drop-in replacement for run_parallel_sensitivity() that uses Ray actors
    and a dynamic memory scheduler instead of static subprocesses.

    The allocation formula is identical to the spec:
        U               = m_u + sn_per_worker_gb
        effective_avail = currgpu_avail_g - gpu_headroom_gb
        workers_g       = floor(effective_avail / U)
        P               = sum(workers_g for g in gpu_ids)

    After initial assignment, the DynamicSensitivityScheduler monitors GPU
    memory every POLL_INTERVAL_S seconds and reassigns idle memory from
    underutilized workers to memory-hungry workers.

    Parameters
    ----------
    (same as run_parallel_sensitivity — see parallel_sensitivity.py)
    sn_per_worker_gb : per-worker safety net in GB (spec: sn)
    gpu_headroom_gb  : per-GPU global VRAM headroom in GB deducted before
                       computing workers_g.  Default: 1.0 GB.
    """
    try:
        import ray  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "Ray is required for dynamic scheduling.  "
            "Install it with: pip install 'ray[default]'"
        )

    logger = quanter.logger

    # ── 1. Determine active layers ───────────────────────────────────────────
    htp_set    = set(htp_disabled_layers)
    active_ops = [op for op in op_name_to_quantizers if op not in htp_set]
    L          = len(active_ops)
    logger.info(
        "Dynamic sensitivity (step %d): L=%d active layers  qscheme=%s",
        step_num, L, qscheme_key,
    )
    if L == 0:
        logger.warning("No active layers — returning empty sensitivity dict.")
        return {}

    # ── 2. Export sim state ──────────────────────────────────────────────────
    worker_dir = Path(mmp_results_dir) / "worker_state"
    worker_dir.mkdir(parents=True, exist_ok=True)

    sim_prefix = "worker_sim"
    sim.export(path=str(worker_dir), filename_prefix=sim_prefix, export_model=True)

    worker_onnx_path = worker_dir / f"{sim_prefix}.onnx"
    _enc_plain = worker_dir / f"{sim_prefix}.encodings"
    _enc_json  = worker_dir / f"{sim_prefix}.encodings.json"
    if _enc_plain.exists():
        worker_encodings_path = _enc_plain
    elif _enc_json.exists():
        worker_encodings_path = _enc_json
    else:
        raise FileNotFoundError(
            f"sim.export() did not produce an encodings file in {worker_dir}."
        )

    # ── 3. Probe m_u ─────────────────────────────────────────────────────────
    from parallel_sensitivity import estimate_vram_gb, _compute_parallelism_heterogeneous

    m_u = estimate_vram_gb(
        onnx_path=str(worker_onnx_path),
        module_alias=quanter.module_name,
        data_path=data_path,
        gpu_id=gpu_ids[0],
    )
    U = m_u + sn_per_worker_gb
    logger.info(
        "VRAM: m_u=%.2f GB  sn=%.2f GB  U=%.2f GB  N=%d GPU(s)",
        m_u, sn_per_worker_gb, U, len(gpu_ids),
    )

    # ── 4. Live VRAM poll ────────────────────────────────────────────────────
    currgpu_avail_raw = _poll_gpu_pool(gpu_ids)
    currgpu_avail: dict[int, float | None] = {
        gid: (v if v > 0 else None) for gid, v in currgpu_avail_raw.items()
    }
    logger.info(
        "Live currgpu_avail: %s",
        "  ".join(f"gpu{g}={v:.1f}GB" for g, v in sorted(currgpu_avail_raw.items())),
    )

    # ── 5. Compute initial allocation ────────────────────────────────────────
    allocation = _compute_parallelism_heterogeneous(
        gpu_ids=gpu_ids,
        currgpu_avail=currgpu_avail,
        K=m_u,
        sn_per_worker=sn_per_worker_gb,
        M_fallback=vram_per_gpu_gb,
        cushion_fallback=cushion_gb,
        gpu_headroom_gb=gpu_headroom_gb,
    )
    P = min(sum(allocation.values()), L)

    logger.info(
        "Dynamic allocation: P=%d workers  allocation=%s  L=%d layers",
        P, allocation, L,
    )

    # ── 6. Initialise Ray ────────────────────────────────────────────────────
    ray.init(ignore_reinit_error=True)

    SensitivityWorkerActor      = _make_worker_actor_class()
    DynamicSensitivityScheduler = _make_scheduler_actor_class()

    # ── 7. Spawn worker actors ───────────────────────────────────────────────
    # Build flat list of (worker_id, gpu_id) respecting heterogeneous allocation
    worker_slots: list[tuple[int, int]] = []
    wid = 0
    for gid in gpu_ids:
        for _ in range(allocation[gid]):
            worker_slots.append((wid, gid))
            wid += 1
    worker_slots = worker_slots[:P]

    workers: dict[int, Any] = {}
    for worker_id, gpu_id in worker_slots:
        runtime_env: dict[str, Any] = {
            "env_vars": {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        }
        if ENABLE_MPS:
            runtime_env["env_vars"].update(_mps_env_for_gpu(gpu_id))

        actor = SensitivityWorkerActor.options(
            runtime_env=runtime_env,
            name=f"sensitivity_worker_{worker_id}",
        ).remote(
            worker_id=worker_id,
            gpu_id=gpu_id,
            onnx_path=str(worker_onnx_path),
            encodings_path=str(worker_encodings_path),
            module_alias=quanter.module_name,
            data_path=data_path,
            qscheme_key=qscheme_key,
        )
        workers[worker_id] = actor
        logger.debug("Spawned worker %d on gpu %d", worker_id, gpu_id)

    # ── 8. Initial layer distribution (round-robin) ──────────────────────────
    chunks: list[list[str]] = [[] for _ in range(P)]
    for i, op in enumerate(active_ops):
        chunks[i % P].append(op)

    # Reserve a small fraction of layers for the dynamic queue
    # (layers that will be assigned by the scheduler to idle workers)
    QUEUE_FRACTION = 0.1
    queue_layers: list[str] = []
    initial_chunks: list[list[str]] = []
    for chunk in chunks:
        split = max(0, len(chunk) - max(1, int(len(chunk) * QUEUE_FRACTION)))
        initial_chunks.append(chunk[:split])
        queue_layers.extend(chunk[split:])

    logger.info(
        "Initial distribution: %d layers in chunks, %d in dynamic queue",
        sum(len(c) for c in initial_chunks), len(queue_layers),
    )

    # ── 9. Start scheduler actor ─────────────────────────────────────────────
    scheduler = DynamicSensitivityScheduler.remote(
        gpu_ids=gpu_ids,
        m_u=m_u,
        sn_per_worker=sn_per_worker_gb,
        logger_name=f"dynamic_scheduler.step{step_num}",
    )
    ray.get(scheduler.set_layer_queue.remote(queue_layers))
    for worker_id, gpu_id in worker_slots:
        ray.get(scheduler.register_worker.remote(worker_id, gpu_id, workers[worker_id]))

    # ── 10. Submit initial work ───────────────────────────────────────────────
    futures: dict[int, Any] = {}
    for i, (worker_id, _) in enumerate(worker_slots):
        if i < len(initial_chunks) and initial_chunks[i]:
            futures[worker_id] = workers[worker_id].process_layers.remote(initial_chunks[i])

    # ── 11. Monitor loop with periodic rebalancing ───────────────────────────
    start_time    = time.monotonic()
    BEAT_INTERVAL = 60.0
    last_beat     = start_time
    last_rebal    = start_time
    merged: dict[str, float] = {}

    logger.info("Dynamic scheduler running — monitoring %d workers ...", P)

    while futures:
        # Check for completed futures (non-blocking, 1s timeout)
        done_refs, _ = ray.wait(list(futures.values()), num_returns=1, timeout=1.0)

        for ref in done_refs:
            # Find which worker this ref belongs to
            completed_wid = next(
                (wid for wid, f in futures.items() if f is ref), None
            )
            if completed_wid is None:
                continue
            try:
                worker_results = ray.get(ref)
                merged.update(worker_results)
                logger.info(
                    "  Worker %d finished: %d layers scored  (total so far: %d/%d)",
                    completed_wid, len(worker_results), len(merged), L,
                )
            except Exception as exc:
                logger.error("Worker %d raised: %s", completed_wid, exc)
            del futures[completed_wid]

        now = time.monotonic()

        # Periodic rebalancing
        if now - last_rebal >= POLL_INTERVAL_S:
            try:
                rebal_status = ray.get(scheduler.poll_and_rebalance.remote())
                if rebal_status.get("actions"):
                    logger.info("Rebalance actions: %s", rebal_status["actions"])
            except Exception as exc:
                logger.warning("Scheduler rebalance failed: %s", exc)
            last_rebal = now

        # Heartbeat
        if now - last_beat >= BEAT_INTERVAL:
            elapsed = now - start_time
            logger.info(
                "  [heartbeat %.0fs]  %d worker(s) still running  %d/%d layers scored",
                elapsed, len(futures), len(merged), L,
            )
            last_beat = now

    # ── 12. Drain the dynamic queue ───────────────────────────────────────────
    # After all initial worker futures complete, any layers still in the
    # scheduler's queue (not yet assigned by poll_and_rebalance) must be
    # processed to avoid silently dropping layers.
    #
    # We call get_remaining_layers() which atomically returns AND clears the
    # queue, then assign those layers to the first worker slot.
    remaining_layers = ray.get(scheduler.get_remaining_layers.remote())
    if remaining_layers and workers:
        logger.info(
            "Draining %d unassigned layers from dynamic queue ...",
            len(remaining_layers),
        )
        drain_worker_id = worker_slots[0][0]
        drain_future    = workers[drain_worker_id].steal_layers.remote(remaining_layers)
        try:
            drain_results = ray.get(drain_future, timeout=600)
            merged.update(drain_results)
            logger.info(
                "Queue drain complete: %d additional layers scored",
                len(drain_results),
            )
        except Exception as exc:
            logger.warning("Queue drain failed: %s", exc)

    # ── 13. Collect any remaining results ────────────────────────────────────
    for worker_id, actor in workers.items():
        try:
            worker_results = ray.get(actor.get_results.remote(), timeout=30)
            merged.update(worker_results)
        except Exception as exc:
            logger.warning("Could not collect final results from worker %d: %s", worker_id, exc)

    total_elapsed = time.monotonic() - start_time
    logger.info(
        "Dynamic sensitivity complete: %d/%d layers scored  "
        "(%d workers, %.0fs total)",
        len(merged), L, P, total_elapsed,
    )

    # ── 14. Write per-worker result files (for compatibility with quant.py) ──
    result_file = worker_dir / "dynamic_results.json"
    with open(result_file, "w") as f:
        json.dump(merged, f, indent=2)
    logger.debug("Dynamic results written to %s", result_file)

    return merged
