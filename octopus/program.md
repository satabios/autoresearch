# Octopus — Development Roadmap for Production QuantSim Integration

## Context

Octopus (phases 1-9, 46 tests passing) provides GPU worker parallelization for PyTorch, ONNX Runtime, and AIMET QuantSim models. The parent directory contains battle-tested production code for parallel ONNX QuantSim sensitivity analysis:

- `parallel_sensitivity.py` — subprocess-based static orchestrator
- `dynamic_scheduler.py` — Ray-based dynamic scheduler with work-stealing
- `sensitivity_worker.py` — per-layer enabling-loop worker

This roadmap documents 8 gaps between octopus and the production code, with file-by-file implementation instructions.

---

## Gap Analysis

| # | Gap | Octopus Today | Production Code | Priority | Octopus File |
|---|-----|--------------|-----------------|----------|--------------|
| 1 | Wrong QuantSim framework | `aimet_torch` adapter (`adapters/quantsim.py`) | `aimet_onnx.QuantizationSimModel` + `sim.export()` + `load_encodings_to_sim(strict=False)` | **P0** | NEW `adapters/onnx_quantsim.py` |
| 2 | Wrong VRAM profiler for ORT | `torch.cuda` peak memory (`profiler.py:15-73`) | subprocess + `nvidia-smi` delta + `_VRAM_PROBE_SAFETY_MULTIPLIER=1.25` | **P0** | `profiler.py` |
| 3 | torch.cuda-only GPU discovery | `torch.cuda.mem_get_info()` (`discovery.py:42`) | `pynvml.nvmlDeviceGetMemoryInfo()` for physical GPU IDs | **P0** | `discovery.py` |
| 4 | No dynamic work-stealing | Round-robin dispatch in `pool.py:103-107` | 90/10 split + 10s pynvml polling + `SPAWN_HEADROOM_FACTOR=1.1` | **P1** | NEW `scheduler.py` |
| 5 | Data-parallel only worker | `worker.run(batch)` returns inference result | `process_layers(layer_names)` runs enabling loop, returns `{layer: sqnr}` | **P1** | `worker.py` |
| 6 | No CUDA init staggering | All workers spawned simultaneously (`pool.py:62-63`) | `WORKER_LAUNCH_STAGGER_SECONDS=5.0` between launches | **P2** | `pool.py` |
| 7 | No OOM detection/recovery | Generic `WorkerOOMError` exception (`exceptions.py:13`) | Log scanning for `BFCArena`/`CUBLAS_STATUS_ALLOC_FAILED` + actionable `sn` suggestion | **P2** | `exceptions.py` |
| 8 | No NVIDIA MPS support | None | `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` env vars per GPU | **P3** | `pool.py` |

---

## P0 — ONNX QuantSim Adapter + Profiler + Discovery

These three must land together: the adapter defines how workers load models, the profiler measures `m_u`, and discovery provides `currgpu_avail`.

### Gap 1: `adapters/onnx_quantsim.py` (NEW)

The existing `QuantSimAdapter` (`adapters/quantsim.py`) targets `aimet_torch` — it serializes `model.state_dict()` and reconstructs via `QuantizationSimModel(model, dummy_input=torch.randn(1))`. The production code uses `aimet_onnx`, which has a fundamentally different serialization path.

**Production pattern** (from `parallel_sensitivity.py:478-497`):

```python
# Export: sim.export() produces .onnx + .encodings files
sim.export(path=str(worker_dir), filename_prefix="worker_sim", export_model=True)

# Worker side (sensitivity_worker.py:352-415):
onnx_model = onnx.load(onnx_path)
sim = QuantizationSimModel(
    model=onnx_model,
    quant_scheme=QuantScheme.post_training_tf,
    config_file="htp_v81",
    param_type=param_type,
    activation_type=activation_type,
    providers=providers,
)
load_encodings_to_sim(sim, encodings_path, strict=False)
```

**Implementation:**

```python
# adapters/onnx_quantsim.py

class OnnxQuantSimAdapter:
    """Adapter for AIMET ONNX QuantizationSimModel.

    Serialization:
        state_bytes() → sim.export() → tar(worker_sim.onnx + worker_sim.encodings)
        from_state_bytes() → untar → onnx.load() → QuantizationSimModel() → load_encodings_to_sim()
    """

    def __init__(self, sim, qscheme_key: str, config_file: str = "htp_v81"):
        self._sim = sim
        self._qscheme_key = qscheme_key
        self._config_file = config_file

    @property
    def model_type_name(self) -> str:
        return "onnx_quantsim"

    def state_bytes(self) -> bytes:
        """Export sim to a temp dir, tar the .onnx + .encodings, return bytes."""
        # 1. sim.export(path=tmpdir, filename_prefix="ws", export_model=True)
        # 2. Find .onnx and .encodings (or .encodings.json) — handle both naming conventions
        # 3. tar.gz both files + metadata JSON (qscheme_key, config_file)
        # 4. Return bytes
        ...

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn) -> "OnnxQuantSimAdapter":
        """Reconstruct from tarball on the worker side."""
        # 1. Untar to temp dir
        # 2. onnx.load(onnx_path)
        # 3. Resolve QSCHEME_TYPES → (param_type, activation_type)
        # 4. QuantizationSimModel(model, quant_scheme, config_file, param_type, activation_type, providers)
        # 5. load_encodings_to_sim(sim, encodings_path, strict=False)
        #    ↑ strict=False is critical — allows bitwidth mismatches between fresh QuantSim
        #      and exported encodings from a mixed-precision state (sensitivity_worker.py:415)
        ...

    def load_to_device(self, device) -> None:
        """No-op for ORT — providers handle device placement at QuantSim construction."""
        pass

    def unload(self) -> None:
        """Delete the ORT session to free GPU memory."""
        if hasattr(self._sim, 'session'):
            del self._sim.session
```

**Key details:**
- AIMET writes either `worker_sim.encodings` or `worker_sim.encodings.json` — handle both (`parallel_sensitivity.py:486-497`)
- `QSCHEME_TYPES` dict maps `"w8a8"` → `(aimet_onnx.int8, aimet_onnx.int8)` etc. (`sensitivity_worker.py:67-73`)
- ORT providers must include `("CUDAExecutionProvider", {"device_id": 0})` — worker always sees device 0 via `CUDA_VISIBLE_DEVICES`
- Register in `adapters/__init__.py` `_ADAPTER_REGISTRY` and update `detect_and_wrap()` to check for `aimet_onnx.QuantizationSimModel` before `aimet_torch`

### Gap 2: Subprocess VRAM Profiler (`profiler.py`)

The current profiler (`profiler.py:15-73`) uses `torch.cuda.reset_peak_memory_stats()` / `torch.cuda.max_memory_allocated()`. This only works for PyTorch models. For ONNX Runtime, ORT allocates VRAM through CUDA directly — torch.cuda APIs report zero.

**Production pattern** (from `parallel_sensitivity.py:119-300`):

```python
_VRAM_PROBE_SAFETY_MULTIPLIER = 1.25  # QuantSim overhead vs plain ORT

def estimate_vram_gb(onnx_path, module_alias, data_path, gpu_id, n_warmup=20) -> float:
    """Subprocess probe:
    1. nvidia-smi --id=<physical_gpu_id> → baseline_mb
    2. ort.InferenceSession(onnx_path, providers=[CUDA])
    3. Verify CUDAExecutionProvider is actually active (not silent CPU fallback)
    4. nvidia-smi → after_load_mb (captures dominant cost: weights + cuDNN workspace)
    5. Run n_warmup forward passes, track peak_mb
    6. delta_gb = (peak_mb - baseline_mb) / 1024 * _VRAM_PROBE_SAFETY_MULTIPLIER
    """
```

**Implementation — add `profile_ort_vram()` alongside existing `profile_model_vram()`:**

```python
# profiler.py — new function

_VRAM_PROBE_SAFETY_MULTIPLIER = 1.25

def profile_ort_vram(
    onnx_path: str,
    sample_feed_fn: Callable,  # () -> dict[str, np.ndarray]
    gpu_id: int,
    n_warmup: int = 20,
) -> VRAMProfile:
    """Subprocess VRAM probe for ONNX models using nvidia-smi deltas.

    Runs in a subprocess to avoid polluting the parent's CUDA context.
    Uses nvidia-smi with physical GPU ID (not CUDA device index).
    Applies 1.25x multiplier to account for QuantSim overhead vs plain ORT.
    Falls back to 4.0 GB if probe fails.
    """
```

**Why subprocess:** The probe must run in an isolated process because loading an ORT CUDA session creates a CUDA context that persists and consumes ~200-400 MB even after the session is deleted. The parent orchestrator would accumulate stale contexts.

**Why nvidia-smi with physical GPU ID:** `nvidia-smi --id=<gpu_id>` uses physical GPU indices and ignores `CUDA_VISIBLE_DEVICES`. The subprocess sets `CUDA_VISIBLE_DEVICES=<gpu_id>` so ORT's `device_id=0` maps to the correct physical GPU, but nvidia-smi must be called with the physical ID (`parallel_sensitivity.py:178-183`).

**Fallback:** Return `VRAMProfile(peak_vram_gb=4.0, ...)` with a warning if the probe fails, matching `parallel_sensitivity.py:300`.

### Gap 3: pynvml GPU Discovery (`discovery.py`)

`discover_gpus()` (`discovery.py:15-57`) uses `torch.cuda.mem_get_info()` (line 42), which reports VRAM from the CUDA runtime's perspective — filtered by `CUDA_VISIBLE_DEVICES` and potentially stale. Production uses pynvml for live physical GPU polling.

**Production pattern** (from `parallel_sensitivity.py:84-112`):

```python
def _get_currgpu_avail_gb(gpu_id: int) -> float | None:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return float(info.free) / (1024.0 ** 3)
```

**Implementation — dual-mode discovery:**

```python
# discovery.py — add pynvml path

def discover_gpus(device_ids=None, use_pynvml=True) -> list[GPUInfo]:
    """Enumerate GPUs. Prefers pynvml for physical GPU IDs and live memory;
    falls back to torch.cuda if pynvml is unavailable."""
    if use_pynvml:
        try:
            return _discover_via_pynvml(device_ids)
        except ImportError:
            _log.info("pynvml not installed — falling back to torch.cuda")
    return _discover_via_torch(device_ids)  # existing implementation
```

**Why pynvml matters:**
- `torch.cuda.mem_get_info()` uses CUDA device indices (filtered by `CUDA_VISIBLE_DEVICES`); pynvml uses physical GPU indices
- pynvml reports truly unallocated VRAM including memory freed by other processes since last poll
- Enables heterogeneous allocation: GPUs with more free VRAM get more workers (`_compute_parallelism_heterogeneous` in `parallel_sensitivity.py:346-386`)

Also add `poll_gpu_memory(gpu_ids) -> dict[int, float]` for use by the dynamic scheduler (Gap 4).

---

## P1 — Dynamic Scheduler + Sensitivity Worker Mode

### Gap 4: Dynamic Work-Stealing Scheduler (NEW `scheduler.py`)

Current octopus uses static round-robin dispatch (`pool.py:103-107`). Production has two scheduling strategies:

**Static** (`parallel_sensitivity.py:574-595`): Pre-partition L layers into P chunks. Simple, no rebalancing.

**Dynamic** (`dynamic_scheduler.py:403-509`): Reserve 10% of layers in a dynamic queue. Poll pynvml every 10s. When a worker's GPU has `free > U * SPAWN_HEADROOM_FACTOR`, steal layers from the queue and assign them.

**Key constants** (from `dynamic_scheduler.py:84-98`):
```python
POLL_INTERVAL_S = 10.0
SPAWN_HEADROOM_FACTOR = 1.1
MEMORY_PRESSURE_THRESHOLD_GB = 0.5
QUEUE_FRACTION = 0.1  # line 705 of dynamic_scheduler.py
```

**Implementation:**

```python
# scheduler.py — new module

class DynamicScheduler:
    """Monitors GPU memory and reassigns work from a global queue to idle workers.

    Algorithm (from dynamic_scheduler.py:461-507):
    1. Poll _poll_gpu_pool(gpu_ids) → {gpu_id: free_gb}
    2. For each worker: if free_gb >= U * SPAWN_HEADROOM_FACTOR, steal layers
    3. extra_slots = max(1, int(free_gb / U))
    4. steal_count = min(extra_slots, len(queue))
    5. Call worker.steal_layers.remote(stolen)
    """

    def __init__(self, gpu_ids, m_u, sn_per_worker, poll_interval_s=10.0): ...
    def set_work_queue(self, items: list): ...
    def poll_and_rebalance(self) -> dict: ...
    def get_remaining(self) -> list: ...
```

**Integration with `core.py`:** Add a `scheduling="static"|"dynamic"` parameter to `Octopus.__init__()`. Static uses current round-robin. Dynamic uses the new `DynamicScheduler`.

### Gap 5: Sensitivity Worker Mode (`worker.py`)

Current `InferenceWorker.run(batch)` runs a single forward pass per call. Production workers (`sensitivity_worker.py:212-297`, `dynamic_scheduler.py:271-335`) run the full enabling loop: disable all quantizers → enable one layer → eval → disable → next layer.

**Production enabling loop** (from `sensitivity_worker.py:212-297`):

```python
def run_enabling_loop(sim, assigned_layers, op_name_to_quantizers, eval_fn, logger):
    # 1. Snapshot: {id(q): (enabled, data_type, bitwidth, op_mode)}
    # 2. Disable ALL quantizers
    # 3. For each layer:
    #      a. Re-enable layer's quantizers from snapshot
    #      b. q.op_mode = OpMode.quantizeDequantize
    #      c. score = eval_fn(sim.session)
    #      d. Disable layer's quantizers
    # 4. Restore all quantizers to snapshot state
    return {layer_name: sqnr_score}
```

**Implementation — add `SensitivityWorker` alongside existing `InferenceWorker`:**

```python
# worker.py — new Ray actor class

@ray.remote(num_gpus=0)
class SensitivityWorker:
    """Ray actor for enabling-loop sensitivity analysis.

    Unlike InferenceWorker (which runs one batch per call), this worker:
    - Loads the full QuantSim model once
    - Receives a list of layer names
    - Runs the enabling loop (snapshot → disable all → enable one → eval → disable)
    - Returns {layer_name: sqnr_score}
    """

    def __init__(self, model_bytes, adapter_cls_name, eval_fn, qscheme_key): ...
    def initialize(self) -> dict: ...
    def process_layers(self, layer_names: list[str]) -> dict[str, float]: ...
    def steal_layers(self, extra: list[str]) -> dict[str, float]: ...
    def get_status(self) -> dict: ...
    def get_results(self) -> dict[str, float]: ...
```

**Key:** `process_layers()` is the worker's main method. It mirrors `SensitivityWorkerActor.process_layers()` from `dynamic_scheduler.py:271-335`. The worker holds the model for its entire lifetime and processes multiple layers sequentially — no model reload between layers.

---

## P2 — Launch Safety + OOM Detection

### Gap 6: CUDA Init Staggering (`pool.py`)

**Problem:** Launching all workers simultaneously causes CUDA init races — every worker on the same GPU competes for cuBLAS handles and BFCArena memory at the same time, triggering `CUBLAS_STATUS_ALLOC_FAILED` even when total VRAM is sufficient.

**Production pattern** (from `parallel_sensitivity.py:66-77`):
```python
WORKER_LAUNCH_STAGGER_SECONDS = 5.0
# With N GPUs and stagger S, same-GPU workers are S×N seconds apart.
# Total extra overhead = (P-1) × S  (e.g. 47 × 5 = 235s for P=48).
```

**Implementation — modify `pool.py:_start_standard()`:**

Currently line 62: `init_refs = [a.initialize.remote() for a in actors_to_init]` fires all inits simultaneously.

Change to:
```python
WORKER_LAUNCH_STAGGER_S = 5.0

# Group actors by GPU, then stagger init within each group
for i, actor in enumerate(actors_to_init):
    actor.initialize.remote()
    if i < len(actors_to_init) - 1:
        time.sleep(WORKER_LAUNCH_STAGGER_S)
```

Or better: stagger only between workers on the *same* GPU. Workers on different GPUs can init in parallel.

### Gap 7: OOM Detection + Actionable Suggestions (`exceptions.py`)

**Problem:** Current `WorkerOOMError` is a bare exception with no recovery guidance.

**Production pattern** (from `parallel_sensitivity.py:697-738`):
```python
# Scan worker logs for OOM signatures
oom_patterns = ["GPU session init failed", "BFCArena", "CUBLAS_STATUS_ALLOC_FAILED"]

# Actionable suggestion
suggested_sn = sn_per_worker_gb + 1.0
logger.error(
    "Workers %s failed due to GPU OOM. "
    "Suggested: --sn-per-worker %.1f (add ~1 GB per OOM worker)",
    oom_workers, suggested_sn,
)
```

**Implementation:**

```python
# exceptions.py — enhance WorkerOOMError

class WorkerOOMError(OctopusError):
    """A worker hit OOM during inference or model loading.

    Attributes:
        worker_id: which worker failed
        gpu_id: which physical GPU
        oom_pattern: which OOM signature was detected
        suggested_sn_gb: recommended sn_per_worker to prevent recurrence
    """
    OOM_PATTERNS = ["BFCArena", "CUBLAS_STATUS_ALLOC_FAILED", "GPU session init failed"]

    def __init__(self, msg, worker_id=None, gpu_id=None, current_sn=0.0):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.suggested_sn_gb = current_sn + 1.0
        super().__init__(
            f"{msg}\n  Suggested: increase sn_per_worker to {self.suggested_sn_gb:.1f} GB "
            f"or reduce parallelism with a larger safety_net_gb."
        )
```

Also add `--no-cpu-fallback` support: production workers (`sensitivity_worker.py:323-326, 363-396`) exit with `rc=1` on GPU OOM instead of falling back to CPU (which runs for hours). The parent detects `rc=1` and suggests increasing `sn`.

### Gap 8: NVIDIA MPS Support (`pool.py`)

**Production pattern** (from `dynamic_scheduler.py:516-529`):
```python
ENABLE_MPS = False  # set True if MPS daemon is running

def _mps_env_for_gpu(gpu_id: int) -> dict[str, str]:
    return {
        "CUDA_MPS_PIPE_DIRECTORY": f"/tmp/nvidia-mps-gpu{gpu_id}",
        "CUDA_MPS_LOG_DIRECTORY":  f"/tmp/nvidia-log-gpu{gpu_id}",
    }
```

**Implementation — modify `pool.py:_start_standard()`:**

When `enable_mps=True`, add MPS env vars to each actor's `runtime_env["env_vars"]` alongside `CUDA_VISIBLE_DEVICES`.

```python
runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": str(alloc.device_id)}}
if self._enable_mps:
    runtime_env["env_vars"].update({
        "CUDA_MPS_PIPE_DIRECTORY": f"/tmp/nvidia-mps-gpu{alloc.device_id}",
        "CUDA_MPS_LOG_DIRECTORY":  f"/tmp/nvidia-log-gpu{alloc.device_id}",
    })
```

---

## API Example: ONNX QuantSim Sensitivity Scan via Octopus

```python
from octopus import Octopus

# calibrated_sim: aimet_onnx.QuantizationSimModel (already calibrated)
# active_layers: list[str] — op names from connected_graph.ordered_ops
# eval_fn: (ort.InferenceSession) -> float (mean SQNR over dataloader)

with Octopus(
    model=calibrated_sim,
    eval_fn=eval_fn,
    safety_net_gb=1.5,           # sn per worker
    gpu_headroom_gb=1.0,         # per-GPU global reservation
    scheduling="dynamic",        # "static" | "dynamic"
    enable_mps=False,            # requires nvidia-cuda-mps-control -d
    stagger_init_s=5.0,          # seconds between same-GPU worker inits
    no_cpu_fallback=True,        # fail fast on OOM instead of CPU fallback
) as octopus:
    # octopus detects OnnxQuantSimAdapter, profiles via subprocess nvidia-smi,
    # discovers GPUs via pynvml, computes heterogeneous allocation
    results: dict[str, float] = octopus.sensitivity_scan(
        layers=active_layers,
        mode="enabling",         # "enabling" | "disabling"
    )

# results = {"MatMul_0": 45.2, "Conv_1": 38.7, ...}  (layer_name → SQNR dB)
```

**Drop-in replacement for production code:**

```python
# Before (parallel_sensitivity.py):
from parallel_sensitivity import run_parallel_sensitivity
results = run_parallel_sensitivity(quanter, sim, ..., sn_per_worker_gb=1.5)

# After (octopus):
from octopus import Octopus
with Octopus(model=sim, eval_fn=eval_fn, safety_net_gb=1.5) as o:
    results = o.sensitivity_scan(layers=active_layers)
```

---

## File Map (updated)

```
src/octopus/
    core.py          -- Add scheduling= param, sensitivity_scan() method
    discovery.py     -- Add pynvml path, poll_gpu_memory(), heterogeneous allocation
    profiler.py      -- Add profile_ort_vram() subprocess probe with nvidia-smi
    pool.py          -- Add CUDA init stagger, MPS env vars
    worker.py        -- Add SensitivityWorker actor (enabling loop)
    scheduler.py     -- NEW: DynamicScheduler (pynvml poll + work-stealing)
    exceptions.py    -- Enhance WorkerOOMError with OOM patterns + sn suggestion
    adapters/
        onnx_quantsim.py  -- NEW: OnnxQuantSimAdapter (sim.export → tar → from_state_bytes)
        quantsim.py       -- KEEP for aimet_torch use case (unchanged)
```

---

## Implementation Order

1. **P0 together:** `adapters/onnx_quantsim.py` + `profiler.py` (profile_ort_vram) + `discovery.py` (pynvml) — these are co-dependent. The adapter can't be tested without the profiler knowing how to measure ORT VRAM, and the profiler needs pynvml-based GPU discovery.

2. **P1 together:** `worker.py` (SensitivityWorker) + `scheduler.py` (DynamicScheduler) — the worker needs the scheduler to assign layers, and the scheduler needs the worker's `process_layers`/`steal_layers` API.

3. **P2 independently:** `pool.py` (stagger) and `exceptions.py` (OOM detection) can land in any order.

4. **P3 last:** MPS support — only relevant when same-GPU multi-worker is validated.

## Testing

```bash
# Unit tests (mocked, no GPU)
pytest tests/ -v                    # existing 46 + new adapter/profiler/scheduler tests

# Integration tests (requires CUDA + AIMET ONNX)
pytest tests/ -v -m gpu             # OnnxQuantSimAdapter round-trip
pytest tests/ -v -m sensitivity     # enabling loop + dynamic scheduler end-to-end
```

**Round-trip fidelity test for OnnxQuantSimAdapter:**
```python
# Verify: original eval score == restored worker eval score (within FP tolerance)
original_score = eval_fn(sim.session)
adapter = OnnxQuantSimAdapter(sim, qscheme_key="w8a8")
restored = OnnxQuantSimAdapter.from_state_bytes(adapter.state_bytes(), eval_fn)
restored_score = eval_fn(restored._sim.session)
assert abs(original_score - restored_score) < 1e-4
```
