from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable, Iterator, Literal, Optional

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
from octopus.scheduler import DynamicScheduler, partition_layers

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
        scheduling: Literal["static", "dynamic"] = "static",
        stagger_init_s: float = 0.0,
        enable_mps: bool = False,
        no_cpu_fallback: bool = False,
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
            scheduling: "static" for round-robin dispatch (default),
                "dynamic" for pynvml-polled work-stealing scheduler.
            stagger_init_s: Seconds to sleep between same-GPU worker inits.
                0 = no stagger (default). Use 5.0 to avoid CUDA init races
                (BFCArena / CUBLAS_STATUS_ALLOC_FAILED) on dense GPU packing.
            enable_mps: Inject CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY
                env vars per GPU. Requires nvidia-cuda-mps-control -d running.
            no_cpu_fallback: If True, raise WorkerOOMError immediately on GPU OOM
                instead of falling back to CPU (which can run for hours).
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
        self._scheduling = scheduling
        self._stagger_init_s = stagger_init_s
        self._enable_mps = enable_mps
        self._no_cpu_fallback = no_cpu_fallback

        self._pool: Optional[WorkerPool] = None
        self._ray_started_by_us = False
        self._profile: Optional[VRAMProfile] = None
        self._plan: Optional[PoolPlan] = None
        self._gpus: Optional[list[GPUInfo]] = None
        self._pending_refs: list[Any] = []   # ObjectRefs from submit()
        self._results: list[Any] = []         # collected by gather()

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
            stagger_init_s=self._stagger_init_s,
            enable_mps=self._enable_mps,
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

    def sensitivity_scan(
        self,
        layers: list[str],
        mode: Literal["enabling", "disabling"] = "enabling",
    ) -> dict[str, float]:
        """Run per-layer quantization sensitivity analysis.

        Spawns SensitivityWorker actors, distributes layers across them,
        runs the enabling loop on each worker, and collects {layer: sqnr} results.

        Args:
            layers: list of op names (from connected_graph.ordered_ops).
            mode: "enabling" (disable all, enable one at a time) or
                  "disabling" (enable all, disable one at a time).
                  Currently only "enabling" is implemented.

        Returns:
            {layer_name: sqnr_score} for all layers.

        Raises:
            ValueError: if mode is not "enabling".
            RuntimeError: if no GPUs are available.
        """
        if mode != "enabling":
            raise ValueError(
                f"mode={mode!r} not yet implemented. Only 'enabling' is supported."
            )

        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(address=self._ray_address, log_to_driver=False)
            self._ray_started_by_us = True
            _log.info("Ray cluster initialized.")

        # Discover GPUs and profile VRAM if not already done
        if self._gpus is None:
            self._gpus = discover_gpus(device_ids=self._gpu_ids)
            _log.info(
                "Discovered %d GPU(s): %s",
                len(self._gpus),
                [(g.name, f"{g.available_vram_gb:.1f}GB free") for g in self._gpus],
            )

        if self._plan is None:
            # For sensitivity scan, use a dummy profile if not yet profiled.
            # The OnnxQuantSimAdapter's profile_ort_vram should be called externally
            # or we use a conservative default.
            from octopus.profiler import profile_ort_vram
            from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter

            if isinstance(self._adapter, OnnxQuantSimAdapter):
                # Need onnx_path from the adapter's sim
                onnx_path = getattr(self._adapter._sim, "model_path", None)
                if onnx_path is None:
                    # Export to get the path
                    import tempfile, os
                    tmpdir = tempfile.mkdtemp()
                    self._adapter._sim.export(
                        path=tmpdir, filename_prefix="_profile_probe", export_model=True
                    )
                    onnx_path = os.path.join(tmpdir, "_profile_probe.onnx")

                profiling_gpu = max(self._gpus, key=lambda g: g.available_vram_gb).device_id
                self._profile = profile_ort_vram(
                    onnx_path=onnx_path,
                    sample_feed_fn=lambda: {},
                    gpu_id=profiling_gpu,
                )
            else:
                # Fallback: use existing torch profiler with a dummy batch
                profiling_gpu = max(self._gpus, key=lambda g: g.available_vram_gb).device_id
                self._profile = profile_model_vram(self._adapter, None, device_id=profiling_gpu)

            self._plan = compute_worker_allocation(
                self._gpus,
                self._profile.peak_vram_gb,
                self._safety_net_gb,
                self._max_workers,
            )
            _log.info(
                "Worker plan: %d worker(s), model VRAM=%.2f GB",
                self._plan.total_workers,
                self._profile.peak_vram_gb,
            )

        # Import SensitivityWorker here to avoid circular import at module level
        from octopus.worker import SensitivityWorker

        num_workers = self._plan.total_workers
        model_bytes = self._adapter.state_bytes()

        # Partition layers: 90% static, 10% dynamic reserve
        chunks, reserve = partition_layers(layers, num_workers)

        # Spawn SensitivityWorker actors
        workers: list[Any] = []
        workers_by_gpu: dict[int, list[Any]] = {}

        for alloc in self._plan.allocations:
            frac_gpu = 1.0 / alloc.num_workers if alloc.num_workers > 0 else 1.0
            gpu_workers: list[Any] = []
            for _ in range(alloc.num_workers):
                actor = SensitivityWorker.options(
                    num_gpus=frac_gpu,
                    runtime_env={
                        "env_vars": {"CUDA_VISIBLE_DEVICES": str(alloc.device_id)}
                    },
                ).remote(
                    model_bytes,
                    self._adapter.model_type_name,
                    self._adapter.eval_fn,
                )
                workers.append(actor)
                gpu_workers.append(actor)
            workers_by_gpu[alloc.device_id] = gpu_workers

        # Initialize all workers
        init_refs = [w.initialize.remote() for w in workers]
        init_results = ray.get(init_refs)
        for result in init_results:
            _log.info("SensitivityWorker ready: %s", result)

        # Set up dynamic scheduler if requested
        scheduler: Optional[DynamicScheduler] = None
        if self._scheduling == "dynamic" and reserve:
            gpu_ids = [alloc.device_id for alloc in self._plan.allocations]
            scheduler = DynamicScheduler(
                gpu_ids=gpu_ids,
                model_vram_gb=self._profile.peak_vram_gb,
                safety_net_gb=self._safety_net_gb,
            )
            scheduler.set_workers(workers_by_gpu)
            scheduler.set_work_queue(reserve)
            scheduler.start()
        elif reserve:
            # Static: distribute reserve round-robin into chunks
            for i, layer in enumerate(reserve):
                chunks[i % num_workers].append(layer)

        # Dispatch static chunks to workers
        process_refs = [
            workers[i].process_layers.remote(chunks[i])
            for i in range(num_workers)
            if chunks[i]
        ]

        # Collect results
        batch_results_list = ray.get(process_refs)
        all_results: dict[str, float] = {}
        for batch_results in batch_results_list:
            all_results.update(batch_results)

        # Stop dynamic scheduler and collect any remaining results
        if scheduler is not None:
            scheduler.stop()
            remaining = scheduler.get_remaining()
            if remaining:
                _log.warning(
                    "%d layer(s) not processed by dynamic scheduler — "
                    "assigning to first worker.",
                    len(remaining),
                )
                leftover_results = ray.get(workers[0].process_layers.remote(remaining))
                all_results.update(leftover_results)

        # Collect any steal_layers results accumulated in workers
        result_refs = [w.get_results.remote() for w in workers]
        worker_results_list = ray.get(result_refs)
        for worker_results in worker_results_list:
            all_results.update(worker_results)

        # Shut down sensitivity workers
        shutdown_refs = [w.shutdown.remote() for w in workers]
        try:
            ray.get(shutdown_refs, timeout=30)
        except Exception:
            _log.warning("Some SensitivityWorkers did not shut down cleanly.")
        for w in workers:
            try:
                ray.kill(w)
            except Exception:
                pass

        _log.info("sensitivity_scan complete: %d layer(s) evaluated.", len(all_results))
        return all_results

    # ------------------------------------------------------------------
    # submit / gather / map  — parallel for-loop API
    # ------------------------------------------------------------------

    def submit(self, batch: Any) -> Any:
        """Submit a single batch to the worker pool (non-blocking).

        Lazily initialises the pool on the first call (using *batch* for
        VRAM profiling, exactly as ``__call__`` does).

        Returns the Ray ``ObjectRef`` so callers can optionally await it
        directly; most callers will just ignore the return value and call
        :meth:`gather` at the end of the loop.

        Usage::

            with Octopus(model=model, eval_fn=eval_fn) as o:
                for i in range(N):
                    o.submit(inputs[i])
                results = o.gather()
        """
        self._ensure_initialized(batch)
        ref = self._pool.submit(batch)
        self._pending_refs.append(ref)
        return ref

    def gather(self) -> list[Any]:
        """Block until all submitted batches complete and return results.

        Results are returned in submission order (same order as the
        :meth:`submit` calls).  Clears the internal pending queue so
        subsequent :meth:`submit` / :meth:`gather` cycles work correctly.

        Returns:
            List of eval_fn results, one per :meth:`submit` call.
        """
        if not self._pending_refs:
            return list(self._results)
        # Pass a copy so callers (and mocks) see the snapshot, not the
        # mutable list that clear() will empty immediately after.
        refs_snapshot = list(self._pending_refs)
        new_results = ray.get(refs_snapshot)
        self._results.extend(new_results)
        self._pending_refs.clear()
        _log.info("gather(): collected %d result(s).", len(new_results))
        return list(self._results)

    def map(self, inputs: Any) -> list[Any]:
        """Parallel map: submit every item in *inputs*, then gather.

        Equivalent to::

            for x in inputs:
                o.submit(x)
            return o.gather()

        Usage::

            with Octopus(model=model, eval_fn=eval_fn) as o:
                results = o.map(dataset)
        """
        for x in inputs:
            self.submit(x)
        return self.gather()

    @property
    def results(self) -> list[Any]:
        """Results collected by the most recent :meth:`gather` call.

        Also populated automatically when the ``with`` block exits if
        there are pending submitted batches.
        """
        return list(self._results)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all workers and Ray resources."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._ray_started_by_us and ray.is_initialized():
            ray.shutdown()
            self._ray_started_by_us = False
        _log.info("Octopus shut down.")

    def __enter__(self) -> "Octopus":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        # Auto-gather any pending submitted batches before shutting down
        if self._pending_refs:
            try:
                self.gather()
            except Exception as e:
                _log.warning("Auto-gather on __exit__ failed: %s", e)
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
