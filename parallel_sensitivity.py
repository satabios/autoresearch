"""
parallel_sensitivity.py — VRAM-aware parallel per-layer sensitivity orchestrator
=================================================================================

Provides run_parallel_sensitivity(), called by quant.py when
--parallel-sensitivity is set.  Splits the L active layers across P worker
subprocesses (P = sum over GPUs of floor(currgpu_avail_g / U)), where:

    m_u  = peak VRAM consumed by one model instance (GB), measured by dry-run
    sn   = per-worker safety net in GB (prevents OOM from activation spikes)
    U    = m_u + sn   (total memory unit per worker — spec formula)
    M    = VRAM per GPU (GB)
    O_g  = VRAM consumed by other processes on GPU g (measured live via pynvml)
    currgpu_avail_g = M - O_g   (available VRAM on GPU g)
    workers_g = floor(currgpu_avail_g / U)   (workers on GPU g)
    P    = sum(workers_g for g in gpu_pool)  total worker instances
    L    = number of active (non-htp-disabled) quantizable layers
    Each worker handles ≈ L/P layers (round-robin split for load balance).

Formula change vs. original implementation
-------------------------------------------
Original: per_gpu = floor((M - cushion) / K)
          cushion was a fixed GPU-level reservation; K = m_u only.

New:      U = m_u + sn   (sn scales with worker count — spec-correct)
          currgpu_avail = M - O  (O measured live via pynvml, not static)
          per_gpu = floor(currgpu_avail / U)

This means the safety net now correctly scales with the number of workers
per GPU, and the available VRAM is measured from the actual driver state
rather than a static parameter.

Backward compatibility
----------------------
The public API (run_parallel_sensitivity signature) is unchanged.
New parameters sn_per_worker_gb and use_live_vram_poll are keyword-only
with safe defaults so existing callers (quant.py, launch_quant.py) work
without modification.
"""

import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path — needed so workers can import pipeline.*
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

log = logging.getLogger(__name__)

# Path to the worker script (same directory as this file)
_WORKER_SCRIPT = Path(__file__).parent / "sensitivity_worker.py"

# ---------------------------------------------------------------------------
# Conservative launch stagger
# ---------------------------------------------------------------------------
# Workers are assigned round-robin across GPUs (worker i → gpu_ids[i % N]).
# Launching all P workers simultaneously causes every worker on the same GPU
# to race for cuBLAS handles / BFCArena memory at the same time, triggering
# CUBLAS_STATUS_ALLOC_FAILED / OOM even when total VRAM is sufficient.
#
# WORKER_LAUNCH_STAGGER_SECONDS is the sleep inserted between consecutive
# worker launches.  With N GPUs and stagger S, workers on the same GPU are
# S × N seconds apart — enough for the previous worker to finish CUDA init
# before the next one starts.
#
# Example: N=4, S=5 → same-GPU workers are 20 s apart.
# Total extra launch overhead = (P-1) × S  (e.g. 47 × 5 = 235 s for P=48).
WORKER_LAUNCH_STAGGER_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Live VRAM polling via pynvml
# ---------------------------------------------------------------------------

def _get_currgpu_avail_gb(gpu_id: int) -> float | None:
    """Return available (free) VRAM on *gpu_id* in GB using pynvml.

    pynvml queries the NVIDIA driver directly and returns the truly
    unallocated VRAM — including memory freed by other processes since
    the last poll.  This is the correct value for currgpu_avail = M - O
    where O is the memory consumed by all other running processes.

    Returns None if pynvml is unavailable or the query fails, so callers
    can fall back to the static (M - cushion) estimate.
    """
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.free) / (1024.0 ** 3)
    except Exception as exc:
        log.debug("pynvml query failed for gpu %d: %s", gpu_id, exc)
        return None


def _poll_all_gpus_avail(gpu_ids: list[int]) -> dict[int, float | None]:
    """Return {gpu_id: free_gb | None} for every GPU in *gpu_ids*.

    None entries indicate that pynvml is unavailable for that GPU; callers
    should substitute the static (M - cushion) fallback.
    """
    return {gid: _get_currgpu_avail_gb(gid) for gid in gpu_ids}


# ---------------------------------------------------------------------------
# VRAM estimation (m_u profiling)
# ---------------------------------------------------------------------------

def _nvidia_smi_used_mb(gpu_id: int) -> float:
    """Return current used GPU memory in MB via nvidia-smi.

    Returns 0.0 on any error so callers can safely compute a delta.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# Multiplier applied to the raw probe measurement to account for the
# difference between a plain ORT session (probe) and a QuantizationSimModel
# (workers), which wraps the session with additional AIMET state and may
# trigger slightly larger CUDA allocations during the enabling loop.
_VRAM_PROBE_SAFETY_MULTIPLIER: float = 1.25


def estimate_vram_gb(
    onnx_path: str,
    module_alias: str,
    data_path: str,
    gpu_id: int,
    n_warmup: int = 20,
) -> float:
    """Dry-run the ONNX model and return estimated peak VRAM in GB (= m_u).

    This is the profiling step for m_u — the memory footprint of one
    quantized model instance.  It spawns a short-lived subprocess
    (CUDA_VISIBLE_DEVICES=<gpu_id>) that:
      1. Loads the ONNX into an ORT CUDAExecutionProvider session.
      2. Verifies CUDAExecutionProvider is actually active (fails loudly if not).
      3. Snapshots VRAM immediately after session load — this captures the
         dominant cost (model weights + cuDNN workspace init), which is also
         when workers OOM during QuantizationSimModel.__init__.
      4. Runs n_warmup forward passes and tracks peak VRAM during inference.
      5. Applies a _VRAM_PROBE_SAFETY_MULTIPLIER to the raw delta to account
         for QuantizationSimModel overhead vs plain ORT.
      6. Prints "VRAM_GB=<value>" to stdout.

    Using a subprocess avoids polluting the parent process's CUDA context.

    Falls back to 4.0 GB if the probe fails for any reason.
    """
    # Inline probe script — executed as `python -c "<script>"` in a subprocess.
    # repr() is used for all string literals so paths with spaces/quotes are safe.
    #
    # IMPORTANT: nvidia-smi uses PHYSICAL GPU indices and ignores
    # CUDA_VISIBLE_DEVICES.  The subprocess has CUDA_VISIBLE_DEVICES=<gpu_id>
    # set so that ORT's CUDAExecutionProvider(device_id=0) maps to physical
    # GPU gpu_id.  But nvidia-smi must be called with --id=<gpu_id> (the
    # physical ID) to measure memory on the correct GPU.  Using --id=0 would
    # always query physical GPU 0 regardless of CUDA_VISIBLE_DEVICES, giving
    # a near-zero delta when gpu_id != 0.
    probe_script = f"""\
import sys, os, subprocess
from pathlib import Path
sys.path.insert(0, {repr(str(_project_root))})

import onnxruntime as ort
from pipeline.data_loader import create_module_dataloader

# Physical GPU ID — passed in from the parent process.
# nvidia-smi --id uses physical indices, NOT CUDA device indices.
_PHYSICAL_GPU_ID = {gpu_id}

def _mem_mb():
    r = subprocess.run(
        ["nvidia-smi", f"--id={{_PHYSICAL_GPU_ID}}",
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0

baseline_mb = _mem_mb()

session = ort.InferenceSession(
    {repr(onnx_path)},
    providers=[("CUDAExecutionProvider", {{"device_id": 0}}), "CPUExecutionProvider"],
)

# ── Verify CUDA is actually active ───────────────────────────────────────
# ORT silently falls back to CPU if CUDA init fails.  Detect this and abort
# so the parent gets a clear error rather than a near-zero VRAM measurement.
active_providers = session.get_providers()
if "CUDAExecutionProvider" not in active_providers:
    print(f"PROBE_ERROR: CUDAExecutionProvider not active (got {{active_providers}}). "
          "Check CUDA installation and LD_LIBRARY_PATH.", file=sys.stderr)
    sys.exit(1)

# ── Snapshot VRAM right after model load ─────────────────────────────────
# Workers OOM during QuantizationSimModel.__init__ (model load + cuDNN
# workspace init), not during inference.  Capturing VRAM here ensures the
# dominant cost is included in the delta even if forward passes use less.
after_load_mb = _mem_mb()
peak_mb = after_load_mb   # track peak from post-load baseline

dl = create_module_dataloader(
    intermediate_data_path={repr(data_path)},
    module_name={repr(module_alias)},
    batch_size=1,
    shuffle=False,
)
input_names = [inp.name for inp in session.get_inputs()]
for i, batch in enumerate(dl):
    if i >= {n_warmup}:
        break
    inp = batch["input"]
    if isinstance(inp, (list, tuple)):
        feed = dict(zip(input_names, [x[0].numpy() for x in inp]))
    else:
        import torch
        v = inp[0]
        feed = {{input_names[0]: v.numpy() if hasattr(v, "numpy") else v}}
    session.run(None, feed)
    cur = _mem_mb()
    if cur > peak_mb:
        peak_mb = cur

# Raw delta = max(post-load VRAM, peak inference VRAM) - baseline
raw_delta_gb = max(0.0, peak_mb - baseline_mb) / 1024.0

# Apply safety multiplier to account for QuantizationSimModel overhead
# vs plain ORT session used in this probe.
multiplier = {_VRAM_PROBE_SAFETY_MULTIPLIER}
delta_gb = raw_delta_gb * multiplier
print(f"VRAM_GB={{delta_gb:.4f}}")
print(f"VRAM_RAW_GB={{raw_delta_gb:.4f}}")
print(f"VRAM_MULTIPLIER={{multiplier:.2f}}")
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        raw_gb = None
        multiplier_used = _VRAM_PROBE_SAFETY_MULTIPLIER
        for line in result.stdout.splitlines():
            if line.startswith("VRAM_RAW_GB="):
                raw_gb = float(line.split("=", 1)[1])
            if line.startswith("VRAM_MULTIPLIER="):
                multiplier_used = float(line.split("=", 1)[1])
            if line.startswith("VRAM_GB="):
                k = float(line.split("=", 1)[1])
                # Floor at 0.5 GB to avoid division-by-zero / absurd P values
                k = max(k, 0.5)
                log.info(
                    "VRAM probe: m_u = %.2f GB  (raw=%.2f GB × %.2fx multiplier, gpu %d)",
                    k, raw_gb if raw_gb is not None else k / multiplier_used,
                    multiplier_used, gpu_id,
                )
                return k
        # Probe ran but didn't print the expected line — log and fall through
        log.warning(
            "VRAM probe did not emit VRAM_GB= line.\n"
            "  stdout: %s\n  stderr: %s",
            result.stdout[:500], result.stderr[:500],
        )
    except subprocess.TimeoutExpired:
        log.warning("VRAM probe timed out — defaulting to 4.0 GB")
    except Exception as exc:
        log.warning("VRAM probe failed (%s) — defaulting to 4.0 GB", exc)

    return 4.0   # conservative default


# ---------------------------------------------------------------------------
# Parallelism calculator — spec-correct U = m_u + sn formula
# ---------------------------------------------------------------------------

def _compute_parallelism(
    N: int,
    M: float,
    cushion: float,
    K: float,
    sn_per_worker: float = 0.0,
) -> int:
    """Return total worker count P = N × floor(currgpu_avail / U).

    Implements the spec formula:
        U               = m_u + sn   (total memory unit per worker)
        currgpu_avail   = M - cushion  (available VRAM after other processes)
        workers_per_gpu = floor(currgpu_avail / U)
        P               = N × workers_per_gpu

    Clamped to at least 1 so we always run at least one worker.

    Parameters
    ----------
    N              : number of GPUs in the pool
    M              : VRAM per GPU in GB
    cushion        : reserved VRAM per GPU in GB (= O, other-process usage)
    K              : m_u — VRAM consumed by one model instance in GB (dry-run)
    sn_per_worker  : safety net in GB added to each worker's allocation
                     (spec: sn).  Default 0.0 preserves backward compatibility
                     with callers that pass cushion as the sole safety margin.
    """
    U = K + sn_per_worker          # total memory unit per worker (spec: U = m_u + sn)
    available_per_gpu = M - cushion  # currgpu_avail = M - O
    if available_per_gpu <= 0:
        log.warning(
            "cushion (%.1f GB) >= M (%.1f GB) — forcing 1 instance per GPU.",
            cushion, M,
        )
        return N
    per_gpu = max(1, math.floor(available_per_gpu / U))
    return N * per_gpu


def _compute_parallelism_heterogeneous(
    gpu_ids: list[int],
    currgpu_avail: dict[int, float],
    K: float,
    sn_per_worker: float = 0.0,
    M_fallback: float = 80.0,
    cushion_fallback: float = 3.0,
    gpu_headroom_gb: float = 1.0,
) -> dict[int, int]:
    """Return {gpu_id: workers_on_this_gpu} using per-GPU live VRAM availability.

    Unlike _compute_parallelism (which assumes all GPUs have the same available
    VRAM), this function uses the live currgpu_avail measurement from pynvml
    to compute a heterogeneous allocation — GPUs with more free VRAM get more
    workers.

    Parameters
    ----------
    gpu_ids          : physical GPU IDs in the pool
    currgpu_avail    : {gpu_id: free_gb} from _poll_all_gpus_avail()
                       None values fall back to M_fallback - cushion_fallback
    K                : m_u — VRAM per model instance in GB
    sn_per_worker    : per-worker safety net in GB (spec: sn)
    M_fallback       : VRAM per GPU to use when pynvml is unavailable
    cushion_fallback : static cushion to use when pynvml is unavailable
    gpu_headroom_gb  : per-GPU global VRAM headroom in GB reserved before
                       computing workers.  Applied as:
                           effective_avail = currgpu_avail - gpu_headroom_gb
                       Covers driver overhead, fragmentation, and processes
                       that start after the pynvml poll.  Default: 1.0 GB.
    """
    U = K + sn_per_worker
    allocation: dict[int, int] = {}
    for gid in gpu_ids:
        avail = currgpu_avail.get(gid)
        if avail is None:
            avail = M_fallback - cushion_fallback
        effective_avail = max(0.0, avail - gpu_headroom_gb)
        workers = max(1, math.floor(effective_avail / U))
        allocation[gid] = workers
    return allocation


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_parallel_sensitivity(
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
    use_live_vram_poll: bool = True,
    gpu_headroom_gb: float = 1.0,
    no_cpu_fallback: bool = True,
) -> dict[str, float]:
    """Parallel per-layer sensitivity scan.

    Splits the L active layers across P worker subprocesses, where P is
    computed using the spec-correct formula:

        U               = m_u + sn_per_worker_gb
        effective_avail = currgpu_avail_g - gpu_headroom_gb
        workers_g       = floor(effective_avail / U)
        P               = sum(workers_g for g in gpu_ids)

    Returns a {layer_name: sqnr_score} dict identical in shape to
    Quanter.run_per_layer_sensitivity(), so quant.py's downstream MMP logic
    requires no changes.

    Parameters
    ----------
    quanter              : Quanter instance (provides onnx_export_path, module_name, logger)
    sim                  : calibrated QuantizationSimModel (exported to temp dir for workers)
    dummy_input          : unused here (kept for API symmetry with run_per_layer_sensitivity)
    full_forward         : unused here (workers build their own callbacks)
    full_eval            : unused here (workers build their own callbacks)
    mmp_results_dir      : directory for worker state files and logs
    gpu_ids              : physical GPU IDs available for workers
    vram_per_gpu_gb      : M — VRAM per GPU in GB (used as fallback when pynvml unavailable)
    cushion_gb           : O — reserved VRAM per GPU in GB (other processes / driver overhead)
                           Used as fallback when pynvml unavailable.
    qscheme_key          : active quantization scheme (e.g. "w8a8")
    op_name_to_quantizers: {op_name: [quantizers]} built in quant.py
    htp_disabled_layers  : ops whose quantizers are all disabled by htp_v81 config
    data_path            : intermediate dataset path for worker dataloaders
    step_num             : MMP chain step index (used only for log messages)
    sn_per_worker_gb     : per-worker safety net in GB (spec: sn).
                           Added to m_u to form U = m_u + sn.
                           Default 0.0 preserves backward compatibility.
    use_live_vram_poll   : if True, poll currgpu_avail via pynvml before
                           computing P.  Falls back to vram_per_gpu_gb -
                           cushion_gb if pynvml is unavailable.
    gpu_headroom_gb      : per-GPU global VRAM headroom in GB deducted before
                           computing workers_g.  Covers driver overhead,
                           fragmentation, and processes that start after the
                           pynvml poll.  Default: 1.0 GB.
    no_cpu_fallback      : if True (default), pass --no-cpu-fallback to every
                           worker so they exit with rc=1 on GPU OOM instead of
                           silently running for hours on CPU.  Set to False only
                           if you explicitly want the slow CPU fallback behaviour.
    """
    logger = quanter.logger

    # ── 1. Determine active layers ───────────────────────────────────────────
    htp_set    = set(htp_disabled_layers)
    active_ops = [op for op in op_name_to_quantizers if op not in htp_set]
    L          = len(active_ops)
    logger.info(
        "Parallel sensitivity (step %d): L=%d active layers  qscheme=%s",
        step_num, L, qscheme_key,
    )

    if L == 0:
        logger.warning("No active layers — returning empty sensitivity dict.")
        return {}

    # ── 2. Export sim state for workers ──────────────────────────────────────
    # Must happen BEFORE the VRAM probe so the probe measures the QDQ-augmented
    # worker_sim.onnx (~5-6 GB) rather than the plain FP32 ONNX (~2.4 GB).
    # Using the wrong model causes P to be over-estimated, leading to OOM kills.
    worker_dir = Path(mmp_results_dir) / "worker_state"
    worker_dir.mkdir(parents=True, exist_ok=True)

    sim_prefix = "worker_sim"
    sim.export(path=str(worker_dir), filename_prefix=sim_prefix, export_model=True)

    # AIMET may write either worker_sim.encodings or worker_sim.encodings.json
    worker_onnx_path = worker_dir / f"{sim_prefix}.onnx"
    _enc_plain = worker_dir / f"{sim_prefix}.encodings"
    _enc_json  = worker_dir / f"{sim_prefix}.encodings.json"
    if _enc_plain.exists():
        worker_encodings_path = _enc_plain
    elif _enc_json.exists():
        worker_encodings_path = _enc_json
    else:
        raise FileNotFoundError(
            f"sim.export() did not produce an encodings file in {worker_dir}. "
            "Expected worker_sim.encodings or worker_sim.encodings.json."
        )
    logger.debug("Exported sim state: onnx=%s  encodings=%s", worker_onnx_path, worker_encodings_path)

    # ── 3. Estimate m_u (VRAM per instance) via dry-run probe ────────────────
    # Probe uses worker_sim.onnx (QDQ-augmented) — the same model each worker
    # will load — so m_u correctly reflects the ~5-6 GB QuantSim footprint.
    m_u = estimate_vram_gb(
        onnx_path=str(worker_onnx_path),
        module_alias=quanter.module_name,
        data_path=data_path,
        gpu_id=gpu_ids[0],
    )
    U = m_u + sn_per_worker_gb   # total memory unit per worker (spec: U = m_u + sn)
    logger.info(
        "VRAM: m_u=%.2f GB  sn=%.2f GB  U=%.2f GB  M=%.0f GB  cushion=%.0f GB  N=%d GPU(s)",
        m_u, sn_per_worker_gb, U, vram_per_gpu_gb, cushion_gb, len(gpu_ids),
    )

    # ── 4. Poll live currgpu_avail via pynvml ─────────────────────────────────
    N = len(gpu_ids)
    currgpu_avail: dict[int, float | None] = {}

    if use_live_vram_poll:
        currgpu_avail = _poll_all_gpus_avail(gpu_ids)
        live_values = {gid: v for gid, v in currgpu_avail.items() if v is not None}
        if live_values:
            logger.info(
                "Live currgpu_avail (pynvml): %s",
                "  ".join(f"gpu{gid}={v:.1f}GB" for gid, v in sorted(live_values.items())),
            )
        else:
            logger.info(
                "pynvml unavailable — falling back to static currgpu_avail = M - cushion = %.1f GB",
                vram_per_gpu_gb - cushion_gb,
            )
    else:
        logger.info(
            "Live VRAM poll disabled — using static currgpu_avail = M - cushion = %.1f GB",
            vram_per_gpu_gb - cushion_gb,
        )

    # ── 5. Compute per-GPU worker allocation (heterogeneous) ─────────────────
    allocation = _compute_parallelism_heterogeneous(
        gpu_ids=gpu_ids,
        currgpu_avail=currgpu_avail,
        K=m_u,
        sn_per_worker=sn_per_worker_gb,
        M_fallback=vram_per_gpu_gb,
        cushion_fallback=cushion_gb,
        gpu_headroom_gb=gpu_headroom_gb,
    )
    P = sum(allocation.values())
    P = min(P, L)   # can't have more workers than layers

    # Log the per-GPU allocation table
    _col = 12
    _alloc_lines = [
        f"┌{'─'*(_col+2)}┬{'─'*(_col+2)}┬{'─'*(_col+2)}┬{'─'*(_col+2)}┬{'─'*(_col+2)}┐",
        f"│ {'GPU':^{_col}} │ {'avail(GB)':^{_col}} │ {'eff_avail(GB)':^{_col}} │ {'U(GB)':^{_col}} │ {'workers':^{_col}} │",
        f"├{'─'*(_col+2)}┼{'─'*(_col+2)}┼{'─'*(_col+2)}┼{'─'*(_col+2)}┼{'─'*(_col+2)}┤",
    ]
    for gid in gpu_ids:
        avail = currgpu_avail.get(gid)
        avail_val = avail if avail is not None else (vram_per_gpu_gb - cushion_gb)
        eff_avail = max(0.0, avail_val - gpu_headroom_gb)
        avail_str = f"{avail_val:.1f}" if avail is not None else f"{avail_val:.1f}*"
        _alloc_lines.append(
            f"│ {gid:^{_col}} │ {avail_str:^{_col}} │ {eff_avail:^{_col}.1f} │ {U:^{_col}.2f} │ {allocation[gid]:^{_col}} │"
        )
    _alloc_lines.append(
        f"└{'─'*(_col+2)}┴{'─'*(_col+2)}┴{'─'*(_col+2)}┴{'─'*(_col+2)}┴{'─'*(_col+2)}┘"
    )
    _alloc_lines.append(f"  * = pynvml unavailable, used M - cushion fallback")
    logger.info(
        "Worker allocation (P=%d total, L=%d layers → ~%d layers/worker):\n%s",
        P, L, math.ceil(L / max(P, 1)), "\n".join(_alloc_lines),
    )

    # ── 6. Partition layers into P chunks (round-robin for load balance) ─────
    # Build a flat ordered list of (gpu_id, local_worker_idx) assignments
    # that respects the heterogeneous per-GPU allocation.
    gpu_worker_slots: list[int] = []
    for gid in gpu_ids:
        gpu_worker_slots.extend([gid] * allocation[gid])
    # Trim to P (after conservative reduction)
    gpu_worker_slots = gpu_worker_slots[:P]

    chunks: list[list[str]] = [[] for _ in range(P)]
    for i, op in enumerate(active_ops):
        chunks[i % P].append(op)

    chunk_files:  list[Path] = []
    result_files: list[Path] = []
    for i, chunk in enumerate(chunks):
        cf = worker_dir / f"chunk_{i}.json"
        rf = worker_dir / f"result_{i}.json"
        with open(cf, "w") as f:
            json.dump(chunk, f)
        chunk_files.append(cf)
        result_files.append(rf)

    # ── 7. Spawn P worker subprocesses (staggered) ───────────────────────────
    # Workers are launched with a WORKER_LAUNCH_STAGGER_SECONDS delay between
    # each spawn so that workers assigned to the same GPU are separated by
    # WORKER_LAUNCH_STAGGER_SECONDS × N seconds — enough for the previous
    # worker to finish CUDA initialisation before the next one starts.
    logger.info(
        "Spawning %d sensitivity worker(s) with %.0fs stagger between launches ...",
        P, WORKER_LAUNCH_STAGGER_SECONDS,
    )
    procs:     list[tuple[int, subprocess.Popen, Any]] = []   # (idx, proc, log_fh)
    log_paths: list[Path] = []

    active_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if chunk]
    for launch_idx, (i, chunk) in enumerate(active_chunks):
        # Use the heterogeneous slot assignment: worker i → gpu_worker_slots[i]
        gpu_id   = gpu_worker_slots[i] if i < len(gpu_worker_slots) else gpu_ids[i % len(gpu_ids)]
        log_path = worker_dir / f"worker_{i}.log"
        log_paths.append(log_path)

        cmd = [
            sys.executable,
            str(_WORKER_SCRIPT),
            "--onnx-path",      str(worker_onnx_path),
            "--encodings-path", str(worker_encodings_path),
            "--module-alias",   quanter.module_name,
            "--data-path",      data_path,
            "--qscheme-key",    qscheme_key,
            "--layers-file",    str(chunk_files[i]),
            "--output-file",    str(result_files[i]),
            "--gpu",            "0",   # device 0 inside the restricted CUDA env
        ]
        if no_cpu_fallback:
            cmd.append("--no-cpu-fallback")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(
            f"# Worker {i}  GPU={gpu_id}  layers={len(chunk)}  "
            f"step={step_num}  qscheme={qscheme_key}\n"
        )
        log_fh.write(
            f"# m_u={m_u:.2f}GB  sn={sn_per_worker_gb:.2f}GB  "
            f"U={U:.2f}GB  currgpu_avail="
            f"{currgpu_avail.get(gpu_id, vram_per_gpu_gb - cushion_gb):.1f}GB\n"
        )
        log_fh.write(f"# CMD: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}\n\n")
        log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append((i, proc, log_fh))
        logger.debug(
            "  Worker %d: gpu=%d  layers=%d  log=%s",
            i, gpu_id, len(chunk), log_path,
        )

        # Stagger: sleep between launches so same-GPU workers don't race for
        # CUDA init resources.  Skip the sleep after the very last worker.
        if launch_idx < len(active_chunks) - 1:
            time.sleep(WORKER_LAUNCH_STAGGER_SECONDS)

    # ── 8. Wait for all workers ───────────────────────────────────────────────
    start_time    = time.monotonic()
    BEAT_INTERVAL = 60          # seconds between heartbeat log lines
    last_beat     = start_time
    remaining     = list(procs)
    finished:     list[tuple[int, int]] = []   # (worker_idx, return_code)

    while remaining:
        still_running = []
        for i, proc, log_fh in remaining:
            rc = proc.poll()
            if rc is None:
                still_running.append((i, proc, log_fh))
            else:
                log_fh.close()
                elapsed = time.monotonic() - start_time
                status  = "OK" if rc == 0 else f"FAILED(rc={rc})"
                logger.info("  Worker %d %s  (%.0fs elapsed)", i, status, elapsed)
                finished.append((i, rc))
        remaining = still_running

        if remaining:
            now = time.monotonic()
            if now - last_beat >= BEAT_INTERVAL:
                elapsed = now - start_time
                logger.info(
                    "  [heartbeat %.0fs]  %d worker(s) still running",
                    elapsed, len(remaining),
                )
                last_beat = now
            time.sleep(2)

    # ── Check for failures ────────────────────────────────────────────────────
    failed = [(i, rc) for i, rc in finished if rc != 0]
    if failed:
        # Scan worker logs to detect GPU OOM fallback messages so we can give
        # an actionable error message instead of a generic "worker failed".
        oom_workers: list[int] = []
        for i, rc in failed:
            lp = log_paths[i] if i < len(log_paths) else None
            if lp and Path(lp).exists():
                try:
                    log_text = Path(lp).read_text(errors="replace")
                    if "GPU session init failed" in log_text or "BFCArena" in log_text or "CUBLAS_STATUS_ALLOC_FAILED" in log_text:
                        oom_workers.append(i)
                except OSError:
                    pass
            logger.error("Worker %d failed (rc=%d) — log: %s", i, rc, lp)

        if oom_workers:
            # Compute a suggested sn_per_worker that would have prevented OOM.
            # Heuristic: add 1 GB per OOM worker to the current sn_per_worker.
            suggested_sn = sn_per_worker_gb + 1.0
            logger.error(
                "Workers %s failed due to GPU OOM (VRAM exhausted during "
                "QuantSim initialisation). The VRAM probe underestimated the "
                "actual per-worker peak usage.\n"
                "  Current:   --sn-per-worker %.1f  --cushion-gb %.1f\n"
                "  Suggested: --sn-per-worker %.1f  (add ~1 GB per OOM worker)\n"
                "  Or reduce parallelism by increasing --cushion-gb.\n"
                "  Worker logs: %s",
                oom_workers,
                sn_per_worker_gb, cushion_gb,
                suggested_sn,
                worker_dir,
            )
        raise RuntimeError(
            f"{len(failed)}/{P} sensitivity worker(s) failed. "
            f"Check worker logs in {worker_dir}"
            + (
                f"\n  {len(oom_workers)} worker(s) failed due to GPU OOM — "
                f"re-run with --sn-per-worker {sn_per_worker_gb + 1.0:.1f} "
                f"to reduce parallelism and prevent OOM."
                if oom_workers else ""
            )
        )

    # ── 9. Merge results ──────────────────────────────────────────────────────
    merged: dict[str, float] = {}
    for i, rf in enumerate(result_files):
        if not rf.exists():
            logger.warning("Result file missing for worker %d: %s", i, rf)
            continue
        with open(rf) as f:
            merged.update(json.load(f))

    total_elapsed = time.monotonic() - start_time
    logger.info(
        "Parallel sensitivity complete: %d/%d layers scored  "
        "(%d workers, %.0fs total)",
        len(merged), L, P, total_elapsed,
    )
    return merged
