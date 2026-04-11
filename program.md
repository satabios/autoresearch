# sconce — Cleanup & GPU-Aware Pruning Integration

## Context

[sconce](https://github.com/satabios/sconce) is an AutoML model compression package (pruning + quantization + NAS) for PyTorch. The local checkout is at `/Users/sathya/Desktop/Projects/sconce` on the `ViT` branch (5 commits ahead of `main`, adds ViT structural pruning for timm/HF/torchvision).

The current sensitivity scan is a **linear scan** — O(modules × sparsity_steps) deepcopies + full evals, single-GPU, no parallelism, fixed equal budget per layer. This plan replaces it with a GPU-aware binary search algorithm (see `pruning_algorithm.md`) while cleaning up accumulated dead code.

### Branches

| Branch | Status | Action |
|--------|--------|--------|
| `main` | Stable, CNN-only | Base for merge |
| `ViT` | ViT pruning + tests, 5 ahead / 0 behind | **Work branch** — all changes go here, then merge to main |
| `attention` | Superseded by ViT | Delete after ViT merges |

---

## Phase A — Cleanup (no behavioral changes)

All changes in `/Users/sathya/Desktop/Projects/sconce/sconce/`. Run tests after each step to confirm no regressions.

### A1. Delete `model_analyzer.py`

Not imported by `__init__.py` or any module. Contains:
- Hardcoded VGG class, string-`eval()` pruning (replaced by DG/MetaPruner)
- Duplicate `get_input_channel_importance` (already on `prune` class)
- Commented-out driver code

**Action:** `git rm sconce/model_analyzer.py`

### A2. Dead code in `sconce.py`

**Remove these attributes from `__init__` (lines ~114-121):**
```python
self.venum_sorted_list = []     # no venum method exists
self.temp_sparsity_list = []    # unreferenced
self.prune_indexes = []         # unreferenced
self.record_prune_indexes = False  # unreferenced
self.layer_idx = 0              # only used by find_instance (being removed)
self.comparison = True          # unreferenced outside init
```

**Remove `evaluate_model` method (lines ~433-459):**
Only called from commented-out code in `quanter.py`. Duplicates `evaluate()`.

**Fix module-level globals (lines ~17-33):**
```python
# REMOVE — contradictory filterwarnings sequence:
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
# KEEP only specific:
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# REMOVE — module-level device/sync (unused, class has self.device):
device = torch.device("cuda" if ...)
torch.cuda.synchronize()

# REMOVE — module-level fixed seeds (affects every import):
random.seed(321)
np.random.seed(432)
torch.manual_seed(223)
# If needed, move into train() or let user control seeding
```

### A3. Dead code in `pruner.py`

**Remove `find_instance` method (lines ~964-999):**
References `self.venum(sparsity)` which does not exist. Dead code from abandoned experiment.

**Remove `venum_cwp` branch in `channel_prune` (lines ~928, 944-957):**
References `self.venum_sorted_list` and `self.prune_mode == "venum_cwp"`. Simplify to only the normal path.

### A4. Run tests

```bash
cd /Users/sathya/Desktop/Projects/sconce
python -m pytest tests/ -v
```

Confirm: `test_cnn_pruning.py`, `test_vit_pruning.py`, `test_hf_vit_pruning.py`, `test_internvit_pruning.py` all pass.

---

## Phase B — Core Algorithm Integration

### B1. New file: `sconce/gpu_manager.py`

Standalone module — no dependency on `prune` or `sconce` classes.

```
gpu_manager.py
├── GPUInfo (dataclass)
│   └── device_id, total_vram, free_vram, usable_vram
├── WorkerPlan (dataclass)
│   └── total_workers, device_assignments, sharded, gpus_per_model
├── estimate_model_vram(model) → int bytes
│   └── params + buffers + 1.15x overhead
├── estimate_data_vram(batch_size, seq_len, dtype) → int bytes
├── estimate_activation_vram(model) → int bytes
│   └── ~30% of model size (inference heuristic)
├── discover_gpus(safety_net_bytes) → list[GPUInfo]
│   └── torch.cuda.mem_get_info per GPU
└── plan_workers(model, batch_size, seq_len, safety_net_gb) → WorkerPlan
    ├── Case 1: model fits on 1 GPU → pack workers
    ├── Case 2: model > 1 GPU → shard across GPUs
    └── CPU fallback → 1 worker
```

Reference implementation: `/Users/sathya/Desktop/Projects/autoresearch/prune_sensitivity.py` sections 1.

### B2. New method on `prune` class: `metric_degraded`

Insert in `pruner.py` above `_scan_single_module` (~line 502):

```python
def metric_degraded(self, pruned_metric, original_metric, budget):
    """Check if pruning caused unacceptable degradation.
    
    Uses self.lower_is_better to determine direction:
      False (default) → accuracy (higher=better), degradation = drop
      True → loss/BPB (lower=better), degradation = increase
    """
    if self.lower_is_better:
        return pruned_metric > original_metric + budget
    else:
        return original_metric - pruned_metric > budget
```

**Replaces:** hardcoded `abs(acc) <= degradation_value / 3` check on current line 569.

### B3. Extract `_prune_at_ratio` helper

Factor lines 516-565 of current `_scan_single_module` into:

```python
def _prune_at_ratio(self, name, sparsity, example_inputs, vit_config,
                    is_attention, framework, i_layer, total_layers):
    """Apply pruning at given sparsity to self.model (assumed fresh deepcopy).
    
    Handles dispatch to:
    - CWP attention path (DG-based head pruning)
    - CWP custom importance path (DG + MagnitudeImportance)
    - CWP MetaPruner path (standard channel pruning)
    - GMP path (fine_grained_prune on named parameters)
    """
```

Used by `_find_max_prune_ratio` in all three search phases. Avoids duplicating the CWP/GMP dispatch logic.

### B4. Replace `_scan_single_module` → `_find_max_prune_ratio`

**Delete** current `_scan_single_module` (lines 503-584). **Replace with:**

```python
def _find_max_prune_ratio(self, original_model, name, budget,
                          example_inputs, vit_config,
                          i_layer, total_layers,
                          is_attention=False, framework=None,
                          dense_model_accuracy=None):
    """Binary search for the most aggressive sparsity within budget.
    
    Three phases:
    1. Quick probe at max aggression (1 eval) — skip if fully pruneable
    2. Coarse scan step=0.2 (~3-5 evals) — find rough failure region
    3. Binary search within that region (~2 evals) — refine to self.search_tol
    
    Returns: (name, best_sparsity, actual_degradation)
    """
```

**Phase 1 — Quick probe:**
```python
self.model = copy.deepcopy(original_model)
self._prune_at_ratio(name, max_sparsity, ...)
acc = self.evaluate(Tqdm=False)
drop = dense_model_accuracy - acc
if not self.metric_degraded(acc, dense_model_accuracy, budget):
    return (name, max_sparsity, drop)
```

**Phase 2 — Coarse scan:**
```python
coarse_ratios = [0.95, 0.75, 0.55, 0.35, 0.15]  # step=0.2
for i, ratio in enumerate(coarse_ratios):
    self.model = copy.deepcopy(original_model)
    self._prune_at_ratio(name, ratio, ...)
    acc = self.evaluate(Tqdm=False)
    if self.metric_degraded(acc, dense_model_accuracy, budget):
        coarse_lo, coarse_hi = ratio, coarse_ratios[i-1] if i > 0 else 1.0
        break
```

**Phase 3 — Binary search:**
```python
fine_lo, fine_hi = coarse_lo, coarse_hi
best = fine_hi
while fine_hi - fine_lo > self.search_tol:
    mid = (fine_lo + fine_hi) / 2
    self.model = copy.deepcopy(original_model)
    self._prune_at_ratio(name, mid, ...)
    acc = self.evaluate(Tqdm=False)
    if not self.metric_degraded(acc, dense_model_accuracy, budget):
        best = mid
        fine_hi = mid
    else:
        fine_lo = mid
return (name, best, actual_cost)
```

**OOM protection:** Wrap each eval in `try/except RuntimeError` to catch CUDA OOM → treat as "degraded" and continue search.

**Attention heads:** Sparsity maps to discrete head counts. The existing `_compute_head_prune_count` (line 303) already handles rounding `int(round(num_heads * sparsity))`. Binary search still works — it just snaps to valid head counts.

### B5. Refactor `sensitivity_scan` to use orchestration strategies

**Keep the existing signature** (backward compatible):
```python
def sensitivity_scan(self, dense_model_accuracy, scan_step=0.05,
                     scan_start=0.1, scan_end=1.0, verbose=True):
```

**Deprecation:** If user passes non-default `scan_step/scan_start/scan_end`, log:
```python
warnings.warn("scan_step/scan_start/scan_end are deprecated with binary search. "
              "Use self.search_tol instead.", DeprecationWarning)
```

**Internal routing based on `self.search_strategy`:**

#### Strategy: `"adaptive"` (default, best quality)

```
remaining_budget = self.degradation_value
groups = collect modules (same as current lines 621-645)
n = len(groups)

for i, (name, module_info) in enumerate(groups):
    per_layer_budget = remaining_budget / (n - i)
    name, ratio, actual_cost = self._find_max_prune_ratio(
        original_model, name, budget=per_layer_budget, ...)
    self.sparsity_dict[name] = ratio
    remaining_budget -= actual_cost
```

Surplus from tolerant layers funds sensitive ones.

#### Strategy: `"parallel"` (best speed)

```
from .gpu_manager import plan_workers
wp = plan_workers(self.model, batch_size, safety_net_gb=self.safety_net_gb)
per_layer_budget = self.degradation_value / n

# For now: sequential with equal budget (true multi-GPU is Phase C)
for name, module_info in groups:
    name, ratio, _ = self._find_max_prune_ratio(
        original_model, name, budget=per_layer_budget, ...)
    self.sparsity_dict[name] = ratio
```

#### Strategy: `"hybrid"` (parallel phase 1 → adaptive phase 2)

```
# Phase 1: equal-budget pass (sequential for now, parallel in Phase C)
per_layer = self.degradation_value / n
results = {}
for name, module_info in groups:
    name, ratio, cost = self._find_max_prune_ratio(
        original_model, name, budget=per_layer, ...)
    results[name] = (ratio, cost)

# Phase 2: redistribute surplus to ceiling groups
surplus = self.degradation_value - sum(c for _, c in results.values())
ceiling = {n: (r, c) for n, (r, c) in results.items()
           if c >= per_layer * 0.95}
if surplus > self.search_tol and ceiling:
    extra = surplus / len(ceiling)
    for name in ceiling:
        name, ratio, cost = self._find_max_prune_ratio(
            original_model, name, budget=per_layer + extra, ...)
        results[name] = (ratio, cost)

self.sparsity_dict = {n: r for n, (r, _) in results.items()}
```

**Preserved from current code:**
- Two-phase structure: Phase 1 MLP/Conv modules, Phase 2 attention modules (lines 648-675)
- `vit_config` detection (line 616)
- `example_inputs` generation (lines 622-623)
- Module collection logic (lines 621-645)
- `self.model = original_model` restoration at end (line 678)

### B6. New attributes in `sconce.py` `__init__`

Add after line 123 (near other pruning attributes):

```python
# GPU-aware sensitivity search settings
self.search_strategy = "adaptive"   # "adaptive" | "parallel" | "hybrid"
self.safety_net_gb = 1.0            # VRAM safety margin per GPU (GB)
self.search_tol = 0.05              # binary search sparsity granularity
self.lower_is_better = False        # True for loss/BPB, False for accuracy
```

### B7. Update `compress()` in `sconce.py`

Add a print line after the existing prune mode print (~line 335):

```python
print(f"Search strategy: {self.search_strategy} (tol={self.search_tol})")
```

No other changes — `compress()` calls `self.sensitivity_scan()` which reads `self.search_strategy` internally.

### B8. Optional: export `gpu_manager` from `__init__.py`

```python
from .gpu_manager import plan_workers, WorkerPlan
```

Deferred — only needed if advanced users want direct VRAM planning.

---

## Phase C — Multi-GPU Parallelism (follow-up)

True parallel execution requires `torch.multiprocessing` with `spawn` start method. Each subprocess gets its own model copy + GPU assignment. Complexities:

1. `self.evaluate` reads `self.dataloader` — DataLoader must use `num_workers=0` in subprocesses
2. Model + data must be serializable for process spawning
3. CUDA context cannot be shared across processes

### C1. Implement `_worker_fn` in `pruner.py`

```python
def _worker_fn(model_state_dict, group_name, dataset, original_metric,
               budget, device_id, model_cls, model_args, ...):
    """Subprocess entry point for parallel sensitivity search."""
    torch.cuda.set_device(device_id)
    model = model_cls(**model_args)
    model.load_state_dict(model_state_dict)
    model.to(f"cuda:{device_id}")
    # Run binary search on this GPU
    ...
```

### C2. Wire `parallel` strategy to ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

wp = plan_workers(self.model, ...)
with ProcessPoolExecutor(max_workers=wp.total_workers) as pool:
    futures = {pool.submit(_worker_fn, ...): name for name, ... in groups}
    for f in as_completed(futures):
        name, ratio, cost = f.result()
        self.sparsity_dict[name] = ratio
```

### C3. Wire `hybrid` strategy Phase 1 to parallel execution

Same as C2, but followed by sequential Phase 2 redistribution.

---

## Eval Complexity Comparison

```
G = groups, S = sparsity steps (default 19)

Before (linear scan):      G × S deepcopies + evals = G × 19
After (binary search):     G × ~5 deepcopies + evals (avg)
After (+ quick probe):     G × 1 (best) to G × 7 (worst)
After (+ parallel W GPUs): wall-clock ÷ W
After (+ adaptive budget): same evals but better ratios (surplus redistribution)
```

---

## Files Summary

| File | Action | Lines changed |
|------|--------|---------------|
| `sconce/model_analyzer.py` | **Delete** | -436 |
| `sconce/sconce.py` | Clean dead code, add 4 new attrs, update compress() | ~-40 / +10 |
| `sconce/pruner.py` | Replace sensitivity scan with binary search, remove dead code | ~-150 / +200 |
| `sconce/gpu_manager.py` | **New file** | ~+120 |
| `sconce/__init__.py` | Optional: export gpu_manager | ~+1 |

---

## Verification

### Unit tests (existing)
```bash
python -m pytest tests/test_cnn_pruning.py -v        # ResNet18 CWP, small CNN GMP
python -m pytest tests/test_vit_pruning.py -v         # timm/torchvision/HF smoke
python -m pytest tests/test_hf_vit_pruning.py -v      # HF head + MLP pruning
python -m pytest tests/test_internvit_pruning.py -v   # InternViT fused QKV
```

### New tests to add
```
tests/test_gpu_manager.py
├── test_estimate_model_vram — known model, check bytes within 20%
├── test_plan_workers_single_gpu — mock mem_get_info, verify worker count
├── test_plan_workers_sharding — mock large model, verify gpus_per_model > 1
├── test_plan_workers_cpu_fallback — no CUDA, verify 1 worker

tests/test_binary_search.py
├── test_quick_probe_skips_search — fully tolerant layer, verify 1 eval
├── test_binary_finds_boundary — toy model with known threshold
├── test_adaptive_redistribution — 2 groups, verify surplus moves
├── test_metric_degraded_directions — lower_is_better True vs False
├── test_backward_compat — old scan_step params trigger deprecation warning
```

### Integration test
```python
from sconce import sconce

s = sconce()
s.model = SmallCNN()
s.dataloader = {"train": train_loader, "test": test_loader}
s.criterion = nn.CrossEntropyLoss()
s.optimizer = optim.Adam(s.model.parameters())
s.scheduler = optim.lr_scheduler.CosineAnnealingLR(s.optimizer, T_max=5)
s.device = torch.device("cuda")
s.prune_mode = "GMP"
s.search_strategy = "adaptive"   # new
s.search_tol = 0.05              # new
s.epochs = 1
s.compress()
# Verify: sparsity_dict populated, model pruned, accuracy within degradation_value
```

### VRAM verification
```python
from sconce.gpu_manager import plan_workers
wp = plan_workers(model, batch_size=64, seq_len=1024)
print(wp)  # Compare total_workers against nvidia-smi free memory
```
