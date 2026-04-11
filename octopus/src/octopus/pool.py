from __future__ import annotations

from typing import Any, Callable, Optional

import ray

from octopus._logging import get_logger
from octopus._types import PoolPlan
from octopus.worker import InferenceWorker, ShardedInferenceWorkerGroup

_log = get_logger()


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
    ) -> None:
        self._plan = pool_plan
        self._model_bytes = model_bytes
        self._adapter_cls_name = adapter_cls_name
        self._eval_fn = eval_fn
        self._workers: list[ray.actor.ActorHandle] = []
        self._round_robin_idx = 0

    def start(self) -> None:
        """Create and initialize all Ray actors."""
        if self._plan.sharding_required:
            self._start_sharded()
        else:
            self._start_standard()

    def _start_standard(self) -> None:
        """Spawn standard (non-sharded) workers."""
        actors_to_init = []

        for alloc in self._plan.allocations:
            frac_gpu = 1.0 / alloc.num_workers if alloc.num_workers > 0 else 1.0
            for _ in range(alloc.num_workers):
                actor = InferenceWorker.options(
                    num_gpus=frac_gpu,
                    runtime_env={
                        "env_vars": {"CUDA_VISIBLE_DEVICES": str(alloc.device_id)}
                    },
                ).remote(
                    self._model_bytes,
                    self._adapter_cls_name,
                    self._eval_fn,
                )
                actors_to_init.append(actor)

        # Initialize all workers in parallel
        init_refs = [a.initialize.remote() for a in actors_to_init]
        results = ray.get(init_refs)

        for actor, result in zip(actors_to_init, results):
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
