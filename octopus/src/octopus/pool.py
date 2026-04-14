from __future__ import annotations

import time
from typing import Any, Callable, Optional

import ray

from octopus._logging import get_logger
from octopus._types import PoolPlan
from octopus.worker import InferenceWorker, ShardedInferenceWorkerGroup

_log = get_logger()

WORKER_LAUNCH_STAGGER_S = 5.0


class WorkerPool:
    """Manages the lifecycle of Ray InferenceWorker actors.

    Spawns actors according to a PoolPlan, assigns fractional GPU resources
    for co-location, and provides round-robin batch submission.
    """

    def __init__(
        self,
        pool_plan: PoolPlan,
        model_bytes: bytes,
        adapter_cls_name: str,
        eval_fn: Callable,
        stagger_init_s: float = 0.0,
        enable_mps: bool = False,
    ) -> None:
        self._plan = pool_plan
        self._model_bytes = model_bytes
        self._adapter_cls_name = adapter_cls_name
        self._eval_fn = eval_fn
        self._stagger_init_s = stagger_init_s
        self._enable_mps = enable_mps
        self._workers: list[ray.actor.ActorHandle] = []
        self._round_robin_idx = 0

    def start(self) -> None:
        """Create and initialize all Ray actors."""
        if self._plan.sharding_required:
            self._start_sharded()
        else:
            self._start_standard()

    def _start_standard(self) -> None:
        """Spawn standard (non-sharded) workers.

        Workers on different GPUs are spawned in parallel.
        Workers on the *same* GPU are staggered by stagger_init_s seconds to
        avoid CUDA init races (BFCArena / CUBLAS_STATUS_ALLOC_FAILED).
        """
        # Group actors by GPU so we can stagger same-GPU inits
        actors_by_gpu: dict[int, list[ray.actor.ActorHandle]] = {}

        for alloc in self._plan.allocations:
            frac_gpu = 1.0 / alloc.num_workers if alloc.num_workers > 0 else 1.0
            env_vars: dict[str, str] = {"CUDA_VISIBLE_DEVICES": str(alloc.device_id)}
            if self._enable_mps:
                env_vars["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-gpu{alloc.device_id}"
                env_vars["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-log-gpu{alloc.device_id}"

            gpu_actors: list[ray.actor.ActorHandle] = []
            for _ in range(alloc.num_workers):
                actor = InferenceWorker.options(
                    num_gpus=frac_gpu,
                    runtime_env={"env_vars": env_vars},
                ).remote(
                    self._model_bytes,
                    self._adapter_cls_name,
                    self._eval_fn,
                )
                gpu_actors.append(actor)
            actors_by_gpu[alloc.device_id] = gpu_actors

        # Fire first worker on each GPU simultaneously, then stagger within each GPU group
        stagger = self._stagger_init_s
        all_actors: list[ray.actor.ActorHandle] = []
        init_refs: list[ray.ObjectRef] = []

        if stagger <= 0:
            # No stagger — init all in parallel (original behaviour)
            for gpu_actors in actors_by_gpu.values():
                for actor in gpu_actors:
                    init_refs.append(actor.initialize.remote())
                    all_actors.append(actor)
        else:
            # Stagger same-GPU workers; different GPUs run in parallel
            # Build per-GPU init sequences, interleaved by slot index
            max_per_gpu = max(len(v) for v in actors_by_gpu.values())
            for slot in range(max_per_gpu):
                slot_actors = [
                    actors[slot]
                    for actors in actors_by_gpu.values()
                    if slot < len(actors)
                ]
                for actor in slot_actors:
                    init_refs.append(actor.initialize.remote())
                    all_actors.append(actor)
                if slot < max_per_gpu - 1:
                    _log.info(
                        "Staggering worker init: sleeping %.1fs before slot %d",
                        stagger,
                        slot + 1,
                    )
                    time.sleep(stagger)

        results = ray.get(init_refs)
        for actor, result in zip(all_actors, results):
            _log.info("Worker ready: %s", result)
            self._workers.append(actor)

        _log.info("Started %d workers.", len(self._workers))

    def _start_sharded(self) -> None:
        """Spawn sharded worker groups."""
        assert self._plan.sharding_strategy is not None
        actors_to_init = []

        for group_rank, alloc in enumerate(self._plan.allocations):
            # For sharded workers, we need the device IDs for the group
            device_ids = list(
                range(alloc.device_id, alloc.device_id + self._plan.gpus_per_shard)
            )
            # Request all GPUs in the shard group
            actor = ShardedInferenceWorkerGroup.options(
                num_gpus=self._plan.gpus_per_shard,
            ).remote(
                self._model_bytes,
                self._adapter_cls_name,
                self._eval_fn,
                device_ids,
                self._plan.sharding_strategy,
                group_rank,
            )
            actors_to_init.append(actor)

        init_refs = [a.initialize.remote() for a in actors_to_init]
        results = ray.get(init_refs)

        for actor, result in zip(actors_to_init, results):
            _log.info("Sharded worker ready: %s", result)
            self._workers.append(actor)

        _log.info("Started %d sharded workers.", len(self._workers))

    def submit(self, batch: Any) -> ray.ObjectRef:
        """Submit a batch to the next worker (round-robin)."""
        worker = self._workers[self._round_robin_idx % len(self._workers)]
        self._round_robin_idx += 1
        return worker.run.remote(batch)

    def shutdown(self) -> None:
        """Gracefully shut down all workers and terminate Ray actors."""
        shutdown_refs = []
        for w in self._workers:
            try:
                shutdown_refs.append(w.shutdown.remote())
            except Exception:
                pass

        if shutdown_refs:
            try:
                ray.get(shutdown_refs, timeout=30)
            except Exception:
                _log.warning("Some workers did not shut down cleanly.")

        for w in self._workers:
            try:
                ray.kill(w)
            except Exception:
                pass

        self._workers.clear()
        _log.info("Worker pool shut down.")

    @property
    def num_workers(self) -> int:
        return len(self._workers)
