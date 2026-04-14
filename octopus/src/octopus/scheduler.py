from __future__ import annotations

import threading
import time
from typing import Any, Optional

from octopus._logging import get_logger
from octopus.discovery import poll_gpu_memory

_log = get_logger()

POLL_INTERVAL_S = 10.0
SPAWN_HEADROOM_FACTOR = 1.1
MEMORY_PRESSURE_THRESHOLD_GB = 0.5
QUEUE_FRACTION = 0.1  # fraction of layers held in dynamic reserve


class DynamicScheduler:
    """Monitors GPU memory and reassigns work from a global queue to idle workers.

    Algorithm (from dynamic_scheduler.py:461-507):
    1. Poll poll_gpu_memory(gpu_ids) → {gpu_id: free_gb}
    2. For each worker: if free_gb >= U * SPAWN_HEADROOM_FACTOR, steal layers
    3. extra_slots = max(1, int(free_gb / U))
    4. steal_count = min(extra_slots, len(queue))
    5. Call worker.steal_layers.remote(stolen)

    Usage::

        scheduler = DynamicScheduler(
            gpu_ids=[0, 1],
            model_vram_gb=2.5,
            safety_net_gb=1.5,
            poll_interval_s=10.0,
        )
        scheduler.set_workers(workers_by_gpu)  # {gpu_id: [actor, ...]}
        scheduler.set_work_queue(reserve_layers)
        scheduler.start()
        # ... workers process their assigned layers ...
        scheduler.stop()
        remaining = scheduler.get_remaining()
    """

    def __init__(
        self,
        gpu_ids: list[int],
        model_vram_gb: float,
        safety_net_gb: float,
        poll_interval_s: float = POLL_INTERVAL_S,
    ) -> None:
        self._gpu_ids = gpu_ids
        self._model_vram_gb = model_vram_gb
        self._safety_net_gb = safety_net_gb
        self._poll_interval_s = poll_interval_s

        self._queue: list[Any] = []
        self._queue_lock = threading.Lock()

        # {gpu_id: [actor_handle, ...]}
        self._workers_by_gpu: dict[int, list[Any]] = {}

        self._steal_refs: list[Any] = []  # pending ray ObjectRefs from steal_layers
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_workers(self, workers_by_gpu: dict[int, list[Any]]) -> None:
        """Register worker actors grouped by physical GPU ID."""
        self._workers_by_gpu = workers_by_gpu

    def set_work_queue(self, items: list[Any]) -> None:
        """Populate the dynamic reserve queue."""
        with self._queue_lock:
            self._queue = list(items)

    def start(self) -> None:
        """Start background polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        _log.info(
            "DynamicScheduler started: %d GPU(s), U=%.2f GB, poll=%.0fs",
            len(self._gpu_ids),
            self._model_vram_gb,
            self._poll_interval_s,
        )

    def stop(self) -> None:
        """Stop background polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_s + 5)
            self._thread = None

    def get_remaining(self) -> list[Any]:
        """Return items still in the dynamic queue (unassigned)."""
        with self._queue_lock:
            return list(self._queue)

    def poll_and_rebalance(self) -> dict:
        """Single poll-and-steal cycle. Returns summary of actions taken."""
        free_by_gpu = poll_gpu_memory(self._gpu_ids)
        actions: dict[int, int] = {}  # gpu_id → layers stolen

        for gpu_id, workers in self._workers_by_gpu.items():
            free_gb = free_by_gpu.get(gpu_id)
            if free_gb is None:
                continue

            threshold = self._model_vram_gb * SPAWN_HEADROOM_FACTOR
            if free_gb < threshold:
                continue

            # How many extra model copies fit in free VRAM?
            extra_slots = max(1, int(free_gb / self._model_vram_gb))

            with self._queue_lock:
                if not self._queue:
                    break
                steal_count = min(extra_slots, len(self._queue))
                stolen = self._queue[:steal_count]
                self._queue = self._queue[steal_count:]

            if not stolen:
                continue

            # Distribute stolen layers across workers on this GPU
            for i, layer in enumerate(stolen):
                worker = workers[i % len(workers)]
                ref = worker.steal_layers.remote([layer])
                self._steal_refs.append(ref)

            actions[gpu_id] = len(stolen)
            _log.info(
                "DynamicScheduler: GPU %d has %.2f GB free (threshold %.2f GB) — "
                "stole %d layer(s) from queue (%d remaining)",
                gpu_id,
                free_gb,
                threshold,
                len(stolen),
                len(self._queue),
            )

        return actions

    def _poll_loop(self) -> None:
        """Background thread: poll every poll_interval_s and rebalance."""
        while self._running:
            try:
                self.poll_and_rebalance()
            except Exception as e:
                _log.warning("DynamicScheduler poll error: %s", e)
            time.sleep(self._poll_interval_s)


def partition_layers(
    layers: list[Any],
    num_workers: int,
    queue_fraction: float = QUEUE_FRACTION,
) -> tuple[list[list[Any]], list[Any]]:
    """Split layers into per-worker static chunks + dynamic reserve queue.

    Args:
        layers: full list of layer names.
        num_workers: number of workers to distribute to.
        queue_fraction: fraction held in dynamic reserve (default 0.1).

    Returns:
        (chunks, reserve) where chunks[i] is the static assignment for worker i
        and reserve is the dynamic queue.
    """
    n_reserve = max(0, int(len(layers) * queue_fraction))
    static_layers = layers[: len(layers) - n_reserve]
    reserve = layers[len(layers) - n_reserve :]

    # Distribute static layers round-robin across workers
    chunks: list[list[Any]] = [[] for _ in range(num_workers)]
    for i, layer in enumerate(static_layers):
        chunks[i % num_workers].append(layer)

    return chunks, reserve
