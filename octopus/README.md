# Octopus

GPU worker parallelization for model inference. Supports PyTorch `nn.Module`, ONNX Runtime `InferenceSession`, and AIMET QuantSim models. Automatically profiles VRAM, discovers GPUs, and distributes work across Ray-managed workers.

---

## Installation

```bash
pip install -e .

# Optional extras
pip install -e ".[onnx]"   # ONNX Runtime GPU support
pip install -e ".[aimet]"  # AIMET Torch QuantSim support
pip install -e ".[dev]"    # pytest, mypy, ruff
pip install -e ".[all]"    # everything
```

---

## Core Concept

```
Before                              After
──────────────────────────────────  ──────────────────────────────────────────
for i in range(N):                  with Octopus(model=model,
    result = eval_fn(model, x[i])       eval_fn=eval_fn) as o:
                                        for i in range(N):
                                            o.submit(x[i])
                                        results = o.gather()
```

Octopus:
1. **Profiles** model VRAM usage on one GPU (dry-run forward pass)
2. **Discovers** all available GPUs via pynvml (or torch.cuda fallback)
3. **Computes** how many workers fit per GPU given a safety-net headroom
4. **Spawns** Ray actors, each owning one model copy
5. **Distributes** batches across workers and collects results

---

## Quick Start

### Pattern 1 — `submit` / `gather` (parallel for-loop)

```python
from octopus import Octopus

with Octopus(model=model, eval_fn=eval_fn) as o:
    for i in range(N):
        o.submit(inputs[i])          # non-blocking
    results = o.gather()             # blocks until all done, ordered list
```

### Pattern 2 — `map` (one-liner)

```python
with Octopus(model=model, eval_fn=eval_fn) as o:
    results = o.map(inputs)          # submit all + gather
```

### Pattern 3 — iterator (streaming)

```python
with Octopus(model=model, eval_fn=eval_fn) as o:
    for result in o(dataset):        # yields results as workers complete
        process(result)
```

### Pattern 4 — reuse pool across loops

```python
o = Octopus(model=model, eval_fn=eval_fn)

# First loop
for x in batch_a:
    o.submit(x)
results_a = o.gather()

# Second loop — same workers, no re-init
for x in batch_b:
    o.submit(x)
results_b = o.gather()

o.shutdown()
```

---

## Constructor Parameters

```python
Octopus(
    model,                          # nn.Module | ort.InferenceSession | QuantSim
    eval_fn,                        # Callable(model, batch) -> result
    safety_net_gb   = 1.0,          # VRAM reserved per GPU (headroom)
    sharding_strategy = "none",     # "tp" | "pp" | "none"
    ordered         = True,         # preserve submission order in results
    gpu_ids         = None,         # restrict to specific GPU indices
    ray_address     = None,         # Ray cluster address (None = local)
    max_workers     = None,         # cap total workers
    prefetch        = 0,            # batches to keep in-flight (0 = 2×workers)
    log_level       = "INFO",
    scheduling      = "static",     # "static" | "dynamic"
    stagger_init_s  = 0.0,          # seconds between same-GPU worker inits
    enable_mps      = False,        # inject CUDA MPS env vars per GPU
    no_cpu_fallback = False,        # fail fast on OOM instead of CPU fallback
)
```

---

## API Reference

### `submit(batch) -> ObjectRef`

Submit a single batch to the worker pool (non-blocking). Lazily initialises the pool on the first call using `batch` for VRAM profiling.

```python
ref = o.submit(my_batch)   # returns Ray ObjectRef (usually ignored)
```

### `gather() -> list`

Block until all submitted batches complete. Returns results in submission order. Clears the internal queue so subsequent `submit`/`gather` cycles work correctly.

```python
results = o.gather()
```

### `map(inputs) -> list`

Parallel map — equivalent to `for x in inputs: submit(x)` then `gather()`.

```python
results = o.map(dataset)
```

### `results` (property)

Results collected by the most recent `gather()` call (or auto-gathered on `__exit__`). Accumulates across multiple `gather()` cycles.

```python
o.results   # list[Any]
```

### `sensitivity_scan(layers, mode="enabling") -> dict[str, float]`

Run per-layer quantization sensitivity analysis for AIMET ONNX QuantSim models.

```python
sqnr_scores = o.sensitivity_scan(
    layers=active_layers,   # list[str] — op names
    mode="enabling",        # "enabling": disable all, enable one at a time
)
# {"MatMul_0": 45.2, "Conv_1": 38.7, ...}
```

---

## Model Adapters

| Model type | Auto-detected | Adapter |
|---|---|---|
| `torch.nn.Module` | ✅ | `PyTorchAdapter` |
| `ort.InferenceSession` | ✅ | `ONNXRuntimeAdapter` |
| `aimet_onnx.QuantizationSimModel` | ✅ | `OnnxQuantSimAdapter` |
| `aimet_torch.QuantizationSimModel` | ✅ | `QuantSimAdapter` |

Adapters handle serialization (`state_bytes()` / `from_state_bytes()`) so model state is shipped to Ray workers without shared memory.

---

## AIMET ONNX QuantSim — Full Example

```python
from octopus import Octopus

# calibrated_sim: aimet_onnx.QuantizationSimModel (already calibrated)
# active_layers:  list[str] — op names from connected_graph.ordered_ops
# eval_fn:        (ort.InferenceSession) -> float  (mean SQNR over dataloader)

with Octopus(
    model=calibrated_sim,
    eval_fn=eval_fn,
    safety_net_gb=1.5,           # VRAM safety net per worker
    gpu_headroom_gb=1.0,         # per-GPU global reservation
    scheduling="dynamic",        # "static" | "dynamic"
    enable_mps=False,            # requires nvidia-cuda-mps-control -d
    stagger_init_s=5.0,          # avoid CUDA init races on dense packing
    no_cpu_fallback=True,        # fail fast on OOM
) as octopus:
    results: dict[str, float] = octopus.sensitivity_scan(
        layers=active_layers,
        mode="enabling",
    )

# results = {"MatMul_0": 45.2, "Conv_1": 38.7, ...}
```

**Drop-in replacement for `parallel_sensitivity.py`:**

```python
# Before:
from parallel_sensitivity import run_parallel_sensitivity
results = run_parallel_sensitivity(quanter, sim, ..., sn_per_worker_gb=1.5)

# After:
from octopus import Octopus
with Octopus(model=sim, eval_fn=eval_fn, safety_net_gb=1.5) as o:
    results = o.sensitivity_scan(layers=active_layers)
```

---

## Scheduling Modes

### Static (default)

Layers are pre-partitioned round-robin across workers. Simple, no rebalancing.

```python
Octopus(..., scheduling="static")
```

### Dynamic (work-stealing)

Reserves 10% of layers in a dynamic queue. A background thread polls GPU free VRAM every 10 s via pynvml. When a GPU has headroom (`free ≥ U × 1.1`), layers are stolen from the queue and assigned to idle workers.

```python
Octopus(..., scheduling="dynamic")
```

---

## GPU Discovery & VRAM Profiling

### GPU Discovery

Octopus prefers **pynvml** for GPU discovery (physical GPU indices, live VRAM readings). Falls back to `torch.cuda` if pynvml is unavailable.

```python
from octopus.discovery import discover_gpus, poll_gpu_memory

gpus = discover_gpus()                    # list[GPUInfo]
free = poll_gpu_memory([0, 1])            # {gpu_id: free_gb}
```

### VRAM Profiling

- **PyTorch / AIMET Torch:** `torch.cuda` peak memory stats
- **ONNX Runtime / AIMET ONNX:** subprocess probe using `nvidia-smi` delta with a 1.25× safety multiplier to account for QuantSim overhead

```python
from octopus.profiler import profile_model_vram, profile_ort_vram
```

---

## Multi-GPU Sharding

For models too large for a single GPU:

```python
# Tensor Parallel — splits weight matrices across GPUs
Octopus(model=model, eval_fn=eval_fn, sharding_strategy="tp")

# Pipeline Parallel — splits layers across GPUs
Octopus(model=model, eval_fn=eval_fn, sharding_strategy="pp")
```

---

## Production Safety Features

### CUDA Init Staggering

Prevents `BFCArena` / `CUBLAS_STATUS_ALLOC_FAILED` races when multiple workers share a GPU:

```python
Octopus(..., stagger_init_s=5.0)   # 5 s between same-GPU worker inits
```

### OOM Detection

`WorkerOOMError` includes the detected OOM pattern and a suggested `safety_net_gb` increase:

```
WorkerOOMError: OOM during inference: BFCArena allocation failed
  Suggested: increase safety_net_gb to 2.5 GB or reduce max_workers.
```

### NVIDIA MPS Support

```python
# Requires: nvidia-cuda-mps-control -d
Octopus(..., enable_mps=True)
# Injects CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY per GPU
```

---

## Testing

```bash
# All unit tests (mocked, no GPU required)
cd octopus
pytest tests/ -v                    # 129 tests

# GPU integration tests
pytest tests/ -v -m gpu             # OnnxQuantSimAdapter round-trip
pytest tests/ -v -m sensitivity     # enabling loop + dynamic scheduler
```

---

## Architecture

```
src/octopus/
    core.py          — Octopus class: submit/gather/map/sensitivity_scan
    discovery.py     — GPU enumeration (pynvml + torch.cuda fallback)
    profiler.py      — VRAM profiling (torch + subprocess nvidia-smi)
    pool.py          — WorkerPool: Ray actor lifecycle, stagger, MPS
    worker.py        — InferenceWorker + SensitivityWorker Ray actors
    scheduler.py     — DynamicScheduler (pynvml poll + work-stealing)
    iterator.py      — OrderedResultIterator / UnorderedResultIterator
    exceptions.py    — WorkerOOMError, InsufficientVRAMError, etc.
    adapters/
        pytorch.py        — PyTorch nn.Module
        onnx_rt.py        — ONNX Runtime InferenceSession
        onnx_quantsim.py  — AIMET ONNX QuantizationSimModel
        quantsim.py       — AIMET Torch QuantizationSimModel
    sharding/
        tensor_parallel.py
        pipeline_parallel.py
