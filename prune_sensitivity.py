"""
GPU-aware pruning sensitivity analysis.

Finds the most aggressive per-group pruning ratio without exceeding
a total metric degradation budget. Uses binary search with quick probing,
coarse-to-fine refinement, adaptive budget reallocation, and VRAM-aware
worker scheduling across multiple GPUs.

Usage:
    ratios = auto_prune_sensitivity(
        model, pruning_groups, dataset,
        threshold=0.1, batch_size=64, seq_len=1024,
        strategy="hybrid",
    )
"""

import math
import torch
from copy import deepcopy
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Any


# ═══════════════════════════════════════════════════════════════
# 1. GPU Resource Manager
# ═══════════════════════════════════════════════════════════════

@dataclass
class GPUInfo:
    device_id: int
    total_vram: int     # bytes
    free_vram: int      # bytes
    usable_vram: int    # free - safety_net


@dataclass
class WorkerPlan:
    total_workers: int
    device_assignments: list  # worker_id → [device_ids]
    sharded: bool             # True if model spans multiple GPUs
    gpus_per_model: int       # 1 for normal, >1 for sharded


def estimate_model_vram(model: torch.nn.Module) -> int:
    """Estimate VRAM for one model copy (params + buffers + allocator overhead)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    overhead = 1.15  # PyTorch allocator fragmentation + metadata
    return int((param_bytes + buffer_bytes) * overhead)


def estimate_data_vram(batch_size: int, seq_len: int, dtype=torch.long) -> int:
    """Estimate VRAM for input + target tensors."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    return 2 * batch_size * seq_len * element_size


def estimate_activation_vram(model: torch.nn.Module) -> int:
    """Conservative estimate of inference activation memory (~30% of model size)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return int(param_bytes * 0.3)


def discover_gpus(safety_net_bytes: int) -> list:
    """Probe all visible GPUs for current VRAM availability."""
    gpus = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        gpus.append(GPUInfo(
            device_id=i,
            total_vram=total,
            free_vram=free,
            usable_vram=max(0, free - safety_net_bytes),
        ))
    return gpus


def plan_workers(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    safety_net_gb: float = 1.0,
) -> WorkerPlan:
    """
    Compute how many parallel workers can run across all GPUs.

    Handles two cases:
      - Model fits on 1 GPU → pack multiple workers per GPU
      - Model too large for 1 GPU → shard across GPUs, 1 worker per shard group
    """
    safety_net = int(safety_net_gb * (1024 ** 3))
    gpus = discover_gpus(safety_net)

    if not gpus:
        raise RuntimeError("No CUDA GPUs available")

    u_model = estimate_model_vram(model)
    u_data = estimate_data_vram(batch_size, seq_len)
    u_act = estimate_activation_vram(model)
    U = u_model + u_data + u_act

    # ── Case 1: model fits on a single GPU ──
    min_usable = min(g.usable_vram for g in gpus)

    if U <= min_usable:
        device_assignments = []
        for gpu in gpus:
            n = gpu.usable_vram // U
            for _ in range(n):
                device_assignments.append([gpu.device_id])

        if not device_assignments:
            device_assignments = [[gpus[0].device_id]]

        return WorkerPlan(
            total_workers=len(device_assignments),
            device_assignments=device_assignments,
            sharded=False,
            gpus_per_model=1,
        )

    # ── Case 2: model must be sharded across GPUs ──
    sorted_gpus = sorted(gpus, key=lambda g: g.usable_vram, reverse=True)
    cumulative = 0
    gpus_needed = 0
    for gpu in sorted_gpus:
        cumulative += gpu.usable_vram
        gpus_needed += 1
        if cumulative >= U:
            break

    if cumulative < U:
        raise RuntimeError(
            f"Model requires {U / 1e9:.1f} GB but all {len(gpus)} GPUs "
            f"only provide {cumulative / 1e9:.1f} GB usable"
        )

    n_shards = len(gpus) // gpus_needed
    device_assignments = []
    for s in range(n_shards):
        start = s * gpus_needed
        device_ids = [sorted_gpus[start + j].device_id for j in range(gpus_needed)]
        device_assignments.append(device_ids)

    return WorkerPlan(
        total_workers=max(1, n_shards),
        device_assignments=device_assignments,
        sharded=True,
        gpus_per_model=gpus_needed,
    )


# ═══════════════════════════════════════════════════════════════
# 2. Metric Degradation Check
# ═══════════════════════════════════════════════════════════════

def metric_degraded(
    pruned: float,
    original: float,
    budget: float,
    lower_is_better: bool = True,
) -> bool:
    """
    Returns True if pruning caused unacceptable degradation.
      lower_is_better=True  → MSE, BPB, loss  (degradation = metric increases)
      lower_is_better=False → SQNR, accuracy   (degradation = metric decreases)
    """
    if lower_is_better:
        return pruned > original + budget
    else:
        return pruned < original - budget


# ═══════════════════════════════════════════════════════════════
# 3. Core: Binary Search with Quick Probe + Coarse-to-Fine
# ═══════════════════════════════════════════════════════════════

def find_max_prune_ratio(
    orig_model: torch.nn.Module,
    group,
    dataset,
    original_metric: float,
    budget: float,
    device,
    prune_fn: Callable,
    eval_fn: Callable,
    lo: float = 0.05,
    hi: float = 1.0,
    tol: float = 0.05,
    lower_is_better: bool = True,
) -> tuple:
    """
    Binary search for the most aggressive keep-ratio within budget.

    ratio = fraction of weights to KEEP (1.0 = no pruning, 0.05 = near-full)

    Args:
        orig_model:  Original (unpruned) model
        group:       Pruning group to test
        dataset:     Evaluation dataset
        original_metric: Baseline metric value
        budget:      Max allowed degradation for this group
        device:      GPU device or list of device ids for sharding
        prune_fn:    Callable(model, group, ratio) → mutates model in-place
        eval_fn:     Callable(model, dataset) → metric value
        lo/hi:       Search bounds for keep-ratio
        tol:         Search stops when hi - lo < tol
        lower_is_better: Metric direction

    Returns:
        (best_ratio, actual_degradation)
    """

    def _test_ratio(ratio):
        test_model = deepcopy(orig_model)
        if isinstance(device, list):
            # Sharded placement — caller provides a model-parallel loader
            for i, dev_id in enumerate(device):
                # Shard placement is model-specific; this is a hook point
                pass
        else:
            test_model = test_model.to(device)
        prune_fn(test_model, group, ratio)
        m = eval_fn(test_model, dataset)
        del test_model
        torch.cuda.empty_cache()
        return m

    # ── Quick probe: test near-full pruning ──
    probe_metric = _test_ratio(lo)
    if not metric_degraded(probe_metric, original_metric, budget, lower_is_better):
        return lo, abs(probe_metric - original_metric)

    # ── Coarse pass: step=0.2, find rough failure region ──
    coarse_step = 0.2
    last_passing_metric = None

    ratio = hi
    coarse_lo, coarse_hi = lo, hi
    while ratio >= lo:
        m = _test_ratio(ratio)
        if metric_degraded(m, original_metric, budget, lower_is_better):
            coarse_lo = ratio
            coarse_hi = ratio + coarse_step
            break
        last_passing_metric = m
        ratio -= coarse_step

    if ratio < lo:
        return lo, abs(last_passing_metric - original_metric)

    # ── Fine pass: binary search within [coarse_lo, coarse_hi] ──
    fine_lo = coarse_lo
    fine_hi = min(coarse_hi, hi)
    best_ratio = fine_hi

    while fine_hi - fine_lo > tol:
        mid = (fine_lo + fine_hi) / 2
        m = _test_ratio(mid)
        if not metric_degraded(m, original_metric, budget, lower_is_better):
            best_ratio = mid
            fine_hi = mid
        else:
            fine_lo = mid

    # Final cost at chosen ratio
    final_metric = _test_ratio(best_ratio)
    actual_cost = abs(final_metric - original_metric)
    return best_ratio, actual_cost


# ═══════════════════════════════════════════════════════════════
# 4. Group-Level Orchestration
# ═══════════════════════════════════════════════════════════════

# ── 4a. Sequential with Adaptive Budget Reallocation ──

def find_all_ratios_adaptive(
    model: torch.nn.Module,
    pruning_groups: list,
    dataset,
    threshold: float,
    prune_fn: Callable,
    eval_fn: Callable,
    lower_is_better: bool = True,
    tol: float = 0.05,
) -> list:
    """
    Sequential processing with adaptive budget redistribution.
    Surplus from tolerant groups is donated to later (more sensitive) groups.
    Best quality, single-GPU.
    """
    orig_model = deepcopy(model)
    original_metric = eval_fn(orig_model, dataset)
    remaining_budget = threshold
    n = len(pruning_groups)
    layer_ratios = []

    for i, G_i in enumerate(pruning_groups):
        groups_left = n - i
        per_layer_budget = remaining_budget / groups_left

        ratio, actual_cost = find_max_prune_ratio(
            orig_model, G_i, dataset, original_metric,
            budget=per_layer_budget,
            device=torch.device("cuda:0"),
            prune_fn=prune_fn, eval_fn=eval_fn,
            lower_is_better=lower_is_better, tol=tol,
        )

        layer_ratios.append(ratio)
        remaining_budget -= actual_cost

    return layer_ratios


# ── 4b. GPU-Aware Parallel (Auto-Scaled Workers) ──

def _worker_fn(orig_model, group, dataset, original_metric,
               budget, device_ids, prune_fn, eval_fn,
               lower_is_better, tol):
    """Subprocess entry point. Runs on assigned GPU(s)."""
    if len(device_ids) == 1:
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = device_ids  # multi-GPU sharded

    return find_max_prune_ratio(
        orig_model, group, dataset, original_metric,
        budget=budget, device=device,
        prune_fn=prune_fn, eval_fn=eval_fn,
        lower_is_better=lower_is_better, tol=tol,
    )


def find_all_ratios_parallel(
    model: torch.nn.Module,
    pruning_groups: list,
    dataset,
    threshold: float,
    batch_size: int,
    seq_len: int,
    prune_fn: Callable,
    eval_fn: Callable,
    lower_is_better: bool = True,
    tol: float = 0.05,
    safety_net_gb: float = 1.0,
) -> list:
    """
    Parallel group evaluation with VRAM-aware auto-scaling.
    Each group gets an equal budget share. Best speed.
    """
    orig_model = deepcopy(model)
    original_metric = eval_fn(orig_model, dataset)
    per_layer_budget = threshold / len(pruning_groups)

    wp = plan_workers(model, batch_size, seq_len, safety_net_gb)
    print(f"[Scheduler] {wp.total_workers} workers across "
          f"{torch.cuda.device_count()} GPUs "
          f"({'sharded' if wp.sharded else 'replicated'}, "
          f"{wp.gpus_per_model} GPU/model)")

    n = len(pruning_groups)
    layer_ratios = [None] * n

    with ProcessPoolExecutor(max_workers=wp.total_workers) as pool:
        futures = {}
        for i, G_i in enumerate(pruning_groups):
            assignment = wp.device_assignments[i % wp.total_workers]
            f = pool.submit(
                _worker_fn,
                orig_model, G_i, dataset, original_metric,
                per_layer_budget, assignment,
                prune_fn, eval_fn, lower_is_better, tol,
            )
            futures[f] = i

        for future in as_completed(futures):
            i = futures[future]
            ratio, _ = future.result()
            layer_ratios[i] = ratio

    return layer_ratios


# ── 4c. Hybrid: Parallel First Pass → Adaptive Refinement ──

def find_all_ratios_hybrid(
    model: torch.nn.Module,
    pruning_groups: list,
    dataset,
    threshold: float,
    batch_size: int,
    seq_len: int,
    prune_fn: Callable,
    eval_fn: Callable,
    lower_is_better: bool = True,
    tol: float = 0.05,
    safety_net_gb: float = 1.0,
) -> list:
    """
    Phase 1 (parallel): equal-budget search across all groups.
    Phase 2 (sequential): redistribute surplus from under-budget groups
             to ceiling groups and re-search them with a larger budget.
    Best balance of speed and quality.
    """
    orig_model = deepcopy(model)
    original_metric = eval_fn(orig_model, dataset)
    n = len(pruning_groups)
    per_layer_budget = threshold / n

    wp = plan_workers(model, batch_size, seq_len, safety_net_gb)
    print(f"[Scheduler] Phase 1: {wp.total_workers} parallel workers")

    # ── Phase 1: parallel with equal budgets ──
    results = [None] * n

    with ProcessPoolExecutor(max_workers=wp.total_workers) as pool:
        futures = {}
        for i, G_i in enumerate(pruning_groups):
            assignment = wp.device_assignments[i % wp.total_workers]
            f = pool.submit(
                _worker_fn,
                orig_model, G_i, dataset, original_metric,
                per_layer_budget, assignment,
                prune_fn, eval_fn, lower_is_better, tol,
            )
            futures[f] = i

        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

    # ── Phase 2: redistribute surplus to ceiling groups ──
    ratios = [r for r, _ in results]
    costs = [c for _, c in results]

    total_used = sum(costs)
    surplus = threshold - total_used

    if surplus > tol:
        ceiling_groups = [
            i for i, (ratio, cost) in enumerate(results)
            if cost >= per_layer_budget * 0.95
        ]
        if ceiling_groups:
            print(f"[Scheduler] Phase 2: redistributing {surplus:.4f} surplus "
                  f"to {len(ceiling_groups)} ceiling groups")
            extra_per_group = surplus / len(ceiling_groups)
            for i in ceiling_groups:
                new_budget = per_layer_budget + extra_per_group
                assignment = wp.device_assignments[i % wp.total_workers]
                dev = torch.device(f"cuda:{assignment[0]}")
                ratio, _ = find_max_prune_ratio(
                    orig_model, pruning_groups[i], dataset, original_metric,
                    new_budget, device=dev,
                    prune_fn=prune_fn, eval_fn=eval_fn,
                    lower_is_better=lower_is_better, tol=tol,
                )
                ratios[i] = ratio

    return ratios


# ═══════════════════════════════════════════════════════════════
# 5. Entry Point
# ═══════════════════════════════════════════════════════════════

def auto_prune_sensitivity(
    model: torch.nn.Module,
    pruning_groups: list,
    dataset,
    threshold: float,
    batch_size: int,
    seq_len: int,
    prune_fn: Callable,
    eval_fn: Callable,
    strategy: str = "hybrid",
    lower_is_better: bool = True,
    tol: float = 0.05,
    safety_net_gb: float = 1.0,
) -> list:
    """
    Full pruning sensitivity analysis.

    Args:
        model:           Model to analyze
        pruning_groups:  List of pruning groups (PG)
        dataset:         Evaluation dataset
        threshold:       Total degradation budget (in metric units)
        batch_size:      Batch size for evaluation
        seq_len:         Sequence length for evaluation
        prune_fn:        Callable(model, group, ratio) — prunes in-place
        eval_fn:         Callable(model, dataset) → float metric
        strategy:        "adaptive" | "parallel" | "hybrid"
        lower_is_better: True for MSE/BPB/loss, False for accuracy/SQNR
        tol:             Keep-ratio search granularity (default 0.05 = 5%)
        safety_net_gb:   VRAM safety margin per GPU in GB

    Returns:
        list of keep-ratios, one per pruning group
    """
    wp = plan_workers(model, batch_size, seq_len, safety_net_gb)
    print(f"[AutoPrune] {len(pruning_groups)} groups | threshold={threshold} "
          f"| strategy={strategy}")
    print(f"[AutoPrune] VRAM plan: {wp.total_workers} workers, "
          f"{wp.gpus_per_model} GPU/model, sharded={wp.sharded}")

    if strategy == "adaptive":
        ratios = find_all_ratios_adaptive(
            model, pruning_groups, dataset, threshold,
            prune_fn, eval_fn, lower_is_better, tol,
        )
    elif strategy == "parallel":
        ratios = find_all_ratios_parallel(
            model, pruning_groups, dataset, threshold,
            batch_size, seq_len,
            prune_fn, eval_fn, lower_is_better, tol, safety_net_gb,
        )
    elif strategy == "hybrid":
        ratios = find_all_ratios_hybrid(
            model, pruning_groups, dataset, threshold,
            batch_size, seq_len,
            prune_fn, eval_fn, lower_is_better, tol, safety_net_gb,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    print(f"[AutoPrune] Done. Ratios: {[f'{r:.2f}' for r in ratios]}")
    return ratios
