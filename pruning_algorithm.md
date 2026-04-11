# Pruning Sensitivity Analysis — Algorithm Design

## Problem

Given a model with **G** pruning groups and a total metric degradation budget **T**,
find the most aggressive keep-ratio per group such that the sum of per-group
degradations stays within **T**.

---

## Original Algorithm (Buggy)

```
For each sensitivity_ratio (0.95 → 0.05):
  For each group G_i:
    Prune G_i on the same model (cumulative!)
    Evaluate
    If degraded: save ratio
  Reset model (shallow copy!)
```

### Bugs Fixed

| # | Bug | Fix |
|---|---|---|
| 1 | Groups pruned cumulatively (not independently) | Swap loops: outer=groups, inner=ratios |
| 2 | `Model = orig_model` is a shallow reference | `deepcopy(orig_model)` per test |
| 3 | `idx=0` wraps to last element in Python | Explicitly return `1.0` (no pruning) |
| 4 | No early stop per group | `break` after finding threshold |
| 5 | No default for fully-tolerant groups | Default to most aggressive ratio |
| 6 | Metric direction hardcoded (`<=`) | Parameterized `lower_is_better` flag |

---

## Final Algorithm — 5 Combined Optimizations

### Optimization Stack

```
┌─────────────────────────────────────────────────────┐
│  5. Entry Point: auto_prune_sensitivity()           │
│     strategy = "adaptive" | "parallel" | "hybrid"   │
├─────────────────────────────────────────────────────┤
│  4. Orchestration                                   │
│     adaptive:  sequential + budget reallocation     │
│     parallel:  VRAM-aware auto-scaled workers       │
│     hybrid:    parallel pass 1 → adaptive pass 2    │
├─────────────────────────────────────────────────────┤
│  3. Per-Group Search: find_max_prune_ratio()        │
│     quick probe → coarse scan (0.2) → binary search │
├─────────────────────────────────────────────────────┤
│  2. Metric Check: metric_degraded()                 │
│     lower_is_better=True  → BPB, MSE, loss          │
│     lower_is_better=False → accuracy, SQNR           │
├─────────────────────────────────────────────────────┤
│  1. GPU Manager: plan_workers()                     │
│     VRAM probe → pack workers or shard model        │
└─────────────────────────────────────────────────────┘
```

---

### 1. GPU Resource Manager

Each sensitivity test needs a **full model copy + data + activations** on GPU.

```
Per-worker cost:   U = u_model + u_data + u_activations

u_model   = Σ(param.numel × param.element_size) × 1.15  (overhead)
u_data    = 2 × batch_size × seq_len × element_size
u_act     = u_model × 0.3                               (inference estimate)
```

**Worker planning:**

```
Case 1 — Model fits on 1 GPU:
    workers_per_gpu = (free_vram - safety_net) // U
    total_workers   = Σ workers_per_gpu across K GPUs

Case 2 — Model larger than 1 GPU:
    gpus_per_model  = ceil(U / per_gpu_usable)
    total_workers   = K // gpus_per_model
```

---

### 2. Per-Group Search (Binary Search + Quick Probe + Coarse-to-Fine)

```
find_max_prune_ratio(group, budget):

  1. QUICK PROBE: test ratio=0.05 (near-full pruning)
     If passes → return 0.05 immediately          [1 eval, done]

  2. COARSE SCAN: step from 1.0 down by 0.2
     Find the 0.2-wide interval where failure occurs
                                                   [~3-5 evals]

  3. BINARY SEARCH: within that 0.2-wide interval
     Bisect until hi - lo < tol (0.05)
                                                   [~2 evals]

  Total: 1 (best) to ~7 (worst) evals per group
  vs. 19 for linear scan
```

**Search direction:**

```
ratio (keep-fraction):  1.0 ◄──────────────── 0.05
                        no pruning            max pruning
                        always passes         usually fails

Binary search finds:    ─────────┐
                        last ratio that passes within budget
```

---

### 3. Orchestration Strategies

#### 3a. Adaptive (Sequential + Budget Reallocation)

```
remaining_budget = T

For each group G_i (sequential):
    per_layer = remaining_budget / groups_left
    ratio, cost = binary_search(G_i, per_layer)
    remaining_budget -= cost          ← only deduct actual cost
```

Tolerant groups use less budget → surplus redistributed to sensitive groups.

#### 3b. Parallel (VRAM-Aware Auto-Scaling)

```
W = plan_workers(model, batch_size, seq_len)
per_layer = T / G                     ← equal share

ProcessPoolExecutor(max_workers=W):
    submit all G groups in parallel
    round-robin assign to device groups
```

#### 3c. Hybrid (Parallel → Adaptive Refinement)

```
Phase 1 (parallel):
    Equal-budget search for all groups
    Collect (ratio, actual_cost) per group

Phase 2 (sequential):
    surplus = T - Σ actual_costs
    ceiling_groups = groups that used ≥95% of their budget
    Redistribute surplus evenly to ceiling groups
    Re-search only those groups with enlarged budget
```

---

## Evaluation Complexity

```
G = groups, W = auto-computed workers

                        Evals (total)    Wall-clock evals
                        ─────────────    ────────────────
Linear scan (original): G × 19           G × 19
Binary search:          G × 5            G × 5
+ Quick probe:          G × 1 to 6       G × 1 to 6
+ Coarse-to-fine:       G × 4 avg        G × 4 avg
+ Parallel (W workers): G × 4            G × 4 / W
+ Hybrid:               G × 4 + C × 4    G × 4 / W + C × 4
                        (C = ceiling groups, typically small)
```

---

## Decision Table

| Scenario | Strategy | Reason |
|---|---|---|
| Small model, few groups | `adaptive` | Best quality, speed is adequate |
| Large model, many groups | `parallel` | Wall-clock ÷ W workers |
| Must maximize pruning ratio quality | `hybrid` | Speed of parallel + budget realloc |
| Model > 1 GPU VRAM | Any | `plan_workers` auto-shards transparently |

---

## Usage

```python
from prune_sensitivity import auto_prune_sensitivity

def my_prune(model, group, ratio):
    """Prune group in-place, keeping `ratio` fraction of weights."""
    ...

def my_eval(model, dataset):
    """Evaluate model, return metric (e.g., BPB)."""
    ...

ratios = auto_prune_sensitivity(
    model=model,
    pruning_groups=pruning_groups,
    dataset=val_dataset,
    threshold=0.1,              # total allowed BPB increase
    batch_size=64,
    seq_len=1024,
    prune_fn=my_prune,
    eval_fn=my_eval,
    strategy="hybrid",          # "adaptive" | "parallel" | "hybrid"
    lower_is_better=True,       # BPB: lower is better
    tol=0.05,                   # 5% ratio granularity
    safety_net_gb=1.0,          # 1 GB VRAM safety margin
)

# ratios[i] = keep-fraction for pruning group i
# e.g., [0.85, 0.45, 0.70, 0.05, ...]
```

---

## Implementation

All code lives in `prune_sensitivity.py`. The `prune_fn` and `eval_fn` are passed
as callables, making the algorithm model-agnostic. For this repo, `eval_fn` can
wrap `evaluate_bpb` from `prepare.py`.
