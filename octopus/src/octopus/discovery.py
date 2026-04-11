from __future__ import annotations

import math
from typing import Literal, Optional

import torch

from octopus._logging import get_logger
from octopus._types import GPUInfo, PoolPlan, WorkerAllocation
from octopus.exceptions import InsufficientVRAMError, NoGPUsFoundError

_log = get_logger()


def discover_gpus(device_ids: Optional[list[int]] = None) -> list[GPUInfo]:
    """Enumerate CUDA GPUs and query their VRAM.

    Args:
        device_ids: Restrict to these CUDA ordinals. None = all visible GPUs.

    Returns:
        List of GPUInfo sorted by device_id.

    Raises:
        NoGPUsFoundError: No CUDA-capable GPUs found.
    """
    if not torch.cuda.is_available():
        raise NoGPUsFoundError("CUDA is not available on this system.")

    count = torch.cuda.device_count()
    if count == 0:
        raise NoGPUsFoundError("No CUDA GPUs detected.")

    ids = device_ids if device_ids is not None else list(range(count))
    gpus: list[GPUInfo] = []

    for dev_id in ids:
        if dev_id >= count:
            _log.warning("Device id %d requested but only %d GPUs available, skipping.", dev_id, count)
            continue
        props = torch.cuda.get_device_properties(dev_id)
        free, total = torch.cuda.mem_get_info(dev_id)
        gpus.append(
            GPUInfo(
                device_id=dev_id,
                name=props.name,
                total_vram_gb=total / (1 << 30),
                available_vram_gb=free / (1 << 30),
                compute_capability=(props.major, props.minor),
            )
        )

    if not gpus:
        raise NoGPUsFoundError("No valid GPUs found for the requested device_ids.")

    gpus.sort(key=lambda g: g.device_id)
    return gpus


def compute_worker_allocation(
    gpus: list[GPUInfo],
    model_vram_gb: float,
    safety_net_gb: float = 1.0,
    max_workers: Optional[int] = None,
) -> PoolPlan:
    """Decide how many workers each GPU should host.

    For each GPU:
        usable = available_vram_gb - safety_net_gb
        workers_here = floor(usable / model_vram_gb)

    Args:
        gpus: discovered GPUInfo list.
        model_vram_gb: U, from profiler (u_model + u_data).
        safety_net_gb: reserved headroom per GPU.
        max_workers: optional cap on total workers.

    Returns:
        PoolPlan describing allocations.

    Raises:
        InsufficientVRAMError: no GPU can fit even one worker.
    """
    allocations: list[WorkerAllocation] = []
    total = 0

    for gpu in gpus:
        usable = gpu.available_vram_gb - safety_net_gb
        if usable <= 0:
            continue
        workers_here = int(usable // model_vram_gb)
        if workers_here <= 0:
            continue
        allocations.append(
            WorkerAllocation(
                device_id=gpu.device_id,
                num_workers=workers_here,
                vram_per_worker_gb=model_vram_gb,
                reserved_safety_gb=safety_net_gb,
            )
        )
        total += workers_here

    if total == 0:
        max_avail = max((g.available_vram_gb for g in gpus), default=0)
        raise InsufficientVRAMError(
            f"Model requires {model_vram_gb:.2f} GB VRAM but the most available "
            f"on any GPU is {max_avail:.2f} GB (after {safety_net_gb:.2f} GB safety net). "
            f"Consider enabling sharding (sharding_strategy='tp' or 'pp')."
        )

    if max_workers is not None and total > max_workers:
        total, allocations = _cap_workers(allocations, max_workers)

    return PoolPlan(
        allocations=allocations,
        total_workers=total,
        sharding_required=False,
        sharding_strategy=None,
        gpus_per_shard=1,
    )


def compute_sharded_allocation(
    gpus: list[GPUInfo],
    model_vram_gb: float,
    safety_net_gb: float,
    strategy: Literal["tp", "pp"],
    max_workers: Optional[int] = None,
) -> PoolPlan:
    """Compute allocation when model requires multi-GPU sharding.

    Determines minimum GPUs per shard, then groups GPUs into shard groups.
    Each group is one logical worker.

    Args:
        gpus: discovered GPUInfo list.
        model_vram_gb: total VRAM needed for the full model.
        safety_net_gb: reserved headroom per GPU.
        strategy: "tp" or "pp".
        max_workers: optional cap on total logical workers.

    Returns:
        PoolPlan with sharding_required=True.

    Raises:
        InsufficientVRAMError: not enough GPUs for even one shard group.
    """
    # Sort GPUs by available VRAM descending for best packing
    sorted_gpus = sorted(gpus, key=lambda g: g.available_vram_gb, reverse=True)

    # Compute per-GPU usable VRAM (use minimum across GPUs for uniform sharding)
    usable_per_gpu = [g.available_vram_gb - safety_net_gb for g in sorted_gpus]
    usable_per_gpu = [u for u in usable_per_gpu if u > 0]

    if not usable_per_gpu:
        raise InsufficientVRAMError("No GPU has usable VRAM after safety net.")

    # For TP: model memory splits ~linearly, add 10% overhead for comm buffers
    # For PP: model memory splits ~linearly, less communication overhead
    overhead = 1.10 if strategy == "tp" else 1.02
    effective_model_gb = model_vram_gb * overhead

    # Minimum GPUs per shard: ceil(model_size / per_gpu_usable)
    # Use the minimum usable VRAM across GPUs in the group for safety
    min_usable = min(usable_per_gpu)
    gpus_per_shard = math.ceil(effective_model_gb / min_usable)

    if gpus_per_shard > len(usable_per_gpu):
        raise InsufficientVRAMError(
            f"Model requires {gpus_per_shard} GPUs for {strategy.upper()} sharding "
            f"but only {len(usable_per_gpu)} usable GPUs available."
        )

    # Group GPUs into shard groups
    num_groups = len(sorted_gpus) // gpus_per_shard
    if max_workers is not None:
        num_groups = min(num_groups, max_workers)

    allocations: list[WorkerAllocation] = []
    for group_idx in range(num_groups):
        start = group_idx * gpus_per_shard
        group_gpus = sorted_gpus[start : start + gpus_per_shard]
        # Use the first GPU's device_id as the representative
        allocations.append(
            WorkerAllocation(
                device_id=group_gpus[0].device_id,
                num_workers=1,  # one logical worker per shard group
                vram_per_worker_gb=model_vram_gb,
                reserved_safety_gb=safety_net_gb,
            )
        )

    return PoolPlan(
        allocations=allocations,
        total_workers=num_groups,
        sharding_required=True,
        sharding_strategy=strategy,
        gpus_per_shard=gpus_per_shard,
    )


def _cap_workers(
    allocations: list[WorkerAllocation], max_workers: int
) -> tuple[int, list[WorkerAllocation]]:
    """Reduce worker counts across allocations to fit within max_workers.

    Distributes the cap proportionally, removing allocations that drop to 0.
    """
    total = sum(a.num_workers for a in allocations)
    if total <= max_workers:
        return total, allocations

    ratio = max_workers / total
    capped: list[WorkerAllocation] = []
    remaining = max_workers

    for i, alloc in enumerate(allocations):
        if i == len(allocations) - 1:
            # Give remainder to last allocation
            n = remaining
        else:
            n = max(1, int(alloc.num_workers * ratio))
            n = min(n, remaining)
        if n <= 0:
            continue
        remaining -= n
        capped.append(
            WorkerAllocation(
                device_id=alloc.device_id,
                num_workers=n,
                vram_per_worker_gb=alloc.vram_per_worker_gb,
                reserved_safety_gb=alloc.reserved_safety_gb,
            )
        )

    return sum(a.num_workers for a in capped), capped
