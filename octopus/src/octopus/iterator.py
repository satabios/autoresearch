from __future__ import annotations

from collections import deque
from typing import Any, Iterator

import ray

from octopus.exceptions import WorkerCrashedError
from octopus.pool import WorkerPool


class OrderedResultIterator:
    """Yields results in the same order batches were submitted.

    Maintains a sliding window of inflight futures, blocking on the oldest
    to preserve ordering while keeping the pipeline full.
    """

    def __init__(
        self,
        pool: WorkerPool,
        dataset: Iterator[Any],
        prefetch: int,
    ) -> None:
        self._pool = pool
        self._dataset = dataset
        self._prefetch = prefetch or (2 * pool.num_workers)
        self._inflight: deque[ray.ObjectRef] = deque()
        self._exhausted = False

    def __iter__(self) -> OrderedResultIterator:
        return self

    def __next__(self) -> Any:
        self._fill_pipeline()

        if not self._inflight:
            raise StopIteration

        oldest_ref = self._inflight.popleft()
        try:
            result = ray.get(oldest_ref)
        except ray.exceptions.RayActorError as e:
            raise WorkerCrashedError(str(e)) from e

        # Backfill one batch
        self._submit_one()
        return result

    def _fill_pipeline(self) -> None:
        while len(self._inflight) < self._prefetch and not self._exhausted:
            self._submit_one()

    def _submit_one(self) -> None:
        try:
            batch = next(self._dataset)
            ref = self._pool.submit(batch)
            self._inflight.append(ref)
        except StopIteration:
            self._exhausted = True


class UnorderedResultIterator:
    """Yields results as soon as any worker finishes (fastest throughput).

    Uses ray.wait() to return the first completed result from the inflight set.
    """

    def __init__(
        self,
        pool: WorkerPool,
        dataset: Iterator[Any],
        prefetch: int,
    ) -> None:
        self._pool = pool
        self._dataset = dataset
        self._prefetch = prefetch or (2 * pool.num_workers)
        self._inflight: list[ray.ObjectRef] = []
        self._exhausted = False

    def __iter__(self) -> UnorderedResultIterator:
        return self

    def __next__(self) -> Any:
        self._fill_pipeline()

        if not self._inflight:
            raise StopIteration

        ready, remaining = ray.wait(self._inflight, num_returns=1)
        self._inflight = remaining

        try:
            result = ray.get(ready[0])
        except ray.exceptions.RayActorError as e:
            raise WorkerCrashedError(str(e)) from e

        # Backfill one batch
        self._submit_one()
        return result

    def _fill_pipeline(self) -> None:
        while len(self._inflight) < self._prefetch and not self._exhausted:
            self._submit_one()

    def _submit_one(self) -> None:
        try:
            batch = next(self._dataset)
            ref = self._pool.submit(batch)
            self._inflight.append(ref)
        except StopIteration:
            self._exhausted = True
