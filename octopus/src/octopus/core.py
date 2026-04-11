from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable, Iterator, Optional

import ray

from octopus._logging import get_logger, setup_logging
from octopus._types import GPUInfo, PoolPlan, ShardingMode, VRAMProfile
from octopus.adapters import detect_and_wrap
from octopus.adapters.base import ModelAdapter
from octopus.discovery import (
    compute_sharded_allocation,
    compute_worker_allocation,
    discover_gpus,
)
from octopus.iterator import OrderedResultIterator, UnorderedResultIterator
from octopus.pool import WorkerPool
from octopus.profiler import pick_profiling_gpu, profile_model_vram

_log = get_logger()


class Octopus:
    """Parallelizes model inference across GPUs.

    Supports PyTorch nn.Module, ONNX Runtime InferenceSession, and
    AIMET QuantizationSimModel. Automatically profiles VRAM usage,
    discovers GPUs, computes optimal worker placement, and distributes
    batches across Ray-managed workers.

    Usage::

        octopus = Octopus(model=model, eval_fn=my_eval)
        for result in octopus(dataset):
            process(result)
        octopus.shutdown()

    Or as a context manager::

        with Octopus(model=model, eval_fn=my_eval) as o:
            for result in o(dataset):
                process(result)
    """

    def __init__(
        self,
        model: Any,
        eval_fn: Callable[[Any, Any], Any],
        safety_net_gb: float = 1.0,
        sharding_strategy: ShardingMode = "none",
        ordered: bool = True,
        gpu_ids: Optional[list[int]] = None,
        ray_address: Optional[str] = None,
        max_workers: Optional[int] = None,
        prefetch: int = 0,
        log_level: str = "INFO",
    ) -> None:
        """
        Args:
            model: PyTorch nn.Module, ORT InferenceSession, or AIMET QuantSim model.
            eval_fn: Callable(model, batch) -> result. Called by each worker.
            safety_net_gb: VRAM to reserve per GPU (default 1.0 GB).
            sharding_strategy: "tp" for tensor parallel, "pp" for pipeline
                parallel, "none" to disable (error if model too large).
            ordered: If True, results yielded in dataset order.
                If False, results yielded as workers complete (faster).
            gpu_ids: Restrict to these CUDA ordinals. None = all GPUs.
            ray_address: Ray cluster address. None = auto-start local cluster.
            max_workers: Cap total workers. None = use all available VRAM.
            prefetch: Batches to keep in flight. 0 = 2 * num_workers.
            log_level: Logging verbosity.
        """
        setup_logging(log_level)

        self._adapter: ModelAdapter = detect_and_wrap(model, eval_fn)
        self._ordered = ordered
        self._prefetch = prefetch
        self._safety_net_gb = safety_net_gb
        self._sharding_strategy = sharding_strategy
        self._max_workers = max_workers
        self._ray_address = ray_address
        self._gpu_ids = gpu_ids

        self._pool: Optional[WorkerPool] = None
        self._ray_started_by_us = False
        self._profile: Optional[VRAMProfile] = None
        self._plan: Optional[PoolPlan] = None
        self._gpus: Optional[list[GPUInfo]] = None

        _log.info(
            "Octopus initialized with %s adapter (sharding=%s, safety_net=%.1f GB)",
            self._adapter.model_type_name,
            self._sharding_strategy,
            self._safety_net_gb,
        )

    def _ensure_initialized(self, sample_batch: Any) -> None:
        """Lazy initialization: discover GPUs, profile VRAM, create worker pool."""
        if self._pool is not None:
            return

        # 1. Discover GPUs
        self._gpus = discover_gpus(device_ids=self._gpu_ids)
        _log.info(
            "Discovered %d GPU(s): %s",
            len(self._gpus),
            [(g.name, f"{g.available_vram_gb:.1f}GB free") for g in self._gpus],
        )

        # 2. Profile VRAM usage
        profiling_gpu = pick_profiling_gpu(self._gpus)
        self._profile = profile_model_vram(
            self._adapter, sample_batch, device_id=profiling_gpu
        )
        _log.info("Model VRAM usage (U): %.2f GB", self._profile.peak_vram_gb)

        # 3. Compute allocation
        if self._sharding_strategy == "none":
            self._plan = compute_worker_allocation(
                self._gpus,
                self._profile.peak_vram_gb,
                self._safety_net_gb,
                self._max_workers,
            )
        else:
            self._plan = compute_sharded_allocation(
                self._gpus,
                self._profile.peak_vram_gb,
                self._safety_net_gb,
                self._sharding_strategy,
                self._max_workers,
            )

        _log.info(
            "Worker plan: %d worker(s) across %d GPU allocation(s), sharding=%s",
            self._plan.total_workers,
            len(self._plan.allocations),
            self._plan.sharding_strategy or "none",
        )
        for alloc in self._plan.allocations:
            _log.info(
                "  GPU %d: %d worker(s), %.2f GB/worker, %.2f GB safety",
                alloc.device_id,
                alloc.num_workers,
                alloc.vram_per_worker_gb,
                alloc.reserved_safety_gb,
            )

        # 4. Init Ray
        if not ray.is_initialized():
            ray.init(address=self._ray_address, log_to_driver=False)
            self._ray_started_by_us = True
            _log.info("Ray cluster initialized.")

        # 5. Create and start worker pool
        model_bytes = self._adapter.state_bytes()
        self._pool = WorkerPool(
            pool_plan=self._plan,
            model_bytes=model_bytes,
            adapter_cls_name=self._adapter.model_type_name,
            eval_fn=self._adapter.eval_fn,
        )
        self._pool.start()

    def __call__(self, dataset: Iterable[Any]) -> Iterator[Any]:
        """Distribute dataset batches across workers and yield results.

        On first call, performs VRAM profiling and spawns workers. Subsequent
        calls reuse the existing worker pool.

        Args:
            dataset: Iterable of batches. Each item is passed to eval_fn.

        Returns:
            Iterator yielding eval_fn results (ordered or unordered).
        """
        dataset_iter = iter(dataset)

        # Peek at first batch for profiling, then put it back
        first_batch = next(dataset_iter)
        self._ensure_initialized(first_batch)

        full_iter = itertools.chain([first_batch], dataset_iter)

        if self._ordered:
            return OrderedResultIterator(self._pool, full_iter, self._prefetch)
        else:
            return UnorderedResultIterator(self._pool, full_iter, self._prefetch)

    def shutdown(self) -> None:
        """Release all workers and Ray resources."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._ray_started_by_us and ray.is_initialized():
            ray.shutdown()
            self._ray_started_by_us = False
        _log.info("Octopus shut down.")

    def __enter__(self) -> Octopus:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()

    @property
    def num_workers(self) -> int:
        return self._pool.num_workers if self._pool else 0

    @property
    def gpu_info(self) -> list[GPUInfo]:
        return list(self._gpus) if self._gpus else []

    @property
    def vram_profile(self) -> Optional[VRAMProfile]:
        return self._profile

    @property
    def pool_plan(self) -> Optional[PoolPlan]:
        return self._plan
