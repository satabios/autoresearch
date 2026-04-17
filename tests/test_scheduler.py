"""Tests for DynamicScheduler and partition_layers."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from octopus.scheduler import (
    QUEUE_FRACTION,
    SPAWN_HEADROOM_FACTOR,
    DynamicScheduler,
    partition_layers,
)


# ---------------------------------------------------------------------------
# partition_layers
# ---------------------------------------------------------------------------

class TestPartitionLayers:
    def test_basic_split(self):
        layers = list(range(100))
        chunks, reserve = partition_layers(layers, num_workers=4, queue_fraction=0.1)
        assert len(reserve) == 10
        assert sum(len(c) for c in chunks) == 90
        assert len(chunks) == 4

    def test_reserve_is_last_fraction(self):
        layers = list(range(20))
        chunks, reserve = partition_layers(layers, num_workers=2, queue_fraction=0.2)
        assert len(reserve) == 4
        assert reserve == layers[16:]

    def test_zero_queue_fraction(self):
        layers = list(range(10))
        chunks, reserve = partition_layers(layers, num_workers=2, queue_fraction=0.0)
        assert reserve == []
        assert sum(len(c) for c in chunks) == 10

    def test_round_robin_distribution(self):
        layers = list(range(9))
        chunks, _ = partition_layers(layers, num_workers=3, queue_fraction=0.0)
        # 9 layers / 3 workers = 3 each
        assert all(len(c) == 3 for c in chunks)

    def test_uneven_distribution(self):
        layers = list(range(10))
        chunks, reserve = partition_layers(layers, num_workers=3, queue_fraction=0.0)
        # 10 layers, 3 workers: [0,3,6,9], [1,4,7], [2,5,8]
        total = sum(len(c) for c in chunks)
        assert total == 10

    def test_single_worker(self):
        layers = ["a", "b", "c"]
        chunks, reserve = partition_layers(layers, num_workers=1, queue_fraction=0.0)
        assert len(chunks) == 1
        assert chunks[0] == ["a", "b", "c"]
        assert reserve == []

    def test_default_queue_fraction(self):
        layers = list(range(100))
        chunks, reserve = partition_layers(layers, num_workers=5)
        assert len(reserve) == int(100 * QUEUE_FRACTION)


# ---------------------------------------------------------------------------
# DynamicScheduler
# ---------------------------------------------------------------------------

class TestDynamicScheduler:
    def _make_scheduler(self, gpu_ids=None, model_vram_gb=2.0, safety_net_gb=1.0):
        if gpu_ids is None:
            gpu_ids = [0, 1]
        return DynamicScheduler(
            gpu_ids=gpu_ids,
            model_vram_gb=model_vram_gb,
            safety_net_gb=safety_net_gb,
            poll_interval_s=0.05,  # fast for tests
        )

    def test_set_work_queue(self):
        sched = self._make_scheduler()
        sched.set_work_queue(["a", "b", "c"])
        assert sched.get_remaining() == ["a", "b", "c"]

    def test_get_remaining_returns_copy(self):
        sched = self._make_scheduler()
        sched.set_work_queue([1, 2, 3])
        remaining = sched.get_remaining()
        remaining.append(99)
        assert sched.get_remaining() == [1, 2, 3]

    def test_poll_and_rebalance_steals_when_free_vram_sufficient(self):
        """When GPU has enough free VRAM, layers should be stolen from queue."""
        sched = self._make_scheduler(gpu_ids=[0], model_vram_gb=2.0)

        mock_worker = MagicMock()
        mock_worker.steal_layers.remote.return_value = MagicMock()
        sched.set_workers({0: [mock_worker]})
        sched.set_work_queue(["L0", "L1", "L2"])

        # GPU 0 has 5 GB free → threshold = 2.0 * 1.1 = 2.2 → should steal
        with patch("octopus.scheduler.poll_gpu_memory", return_value={0: 5.0}):
            actions = sched.poll_and_rebalance()

        assert 0 in actions
        assert actions[0] > 0
        assert len(sched.get_remaining()) < 3

    def test_poll_and_rebalance_no_steal_when_vram_low(self):
        """When GPU has insufficient free VRAM, nothing should be stolen."""
        sched = self._make_scheduler(gpu_ids=[0], model_vram_gb=2.0)

        mock_worker = MagicMock()
        sched.set_workers({0: [mock_worker]})
        sched.set_work_queue(["L0", "L1"])

        # GPU 0 has 1 GB free → threshold = 2.0 * 1.1 = 2.2 → below threshold
        with patch("octopus.scheduler.poll_gpu_memory", return_value={0: 1.0}):
            actions = sched.poll_and_rebalance()

        assert actions == {}
        assert sched.get_remaining() == ["L0", "L1"]

    def test_poll_and_rebalance_empty_queue_no_steal(self):
        sched = self._make_scheduler(gpu_ids=[0], model_vram_gb=2.0)
        mock_worker = MagicMock()
        sched.set_workers({0: [mock_worker]})
        sched.set_work_queue([])

        with patch("octopus.scheduler.poll_gpu_memory", return_value={0: 10.0}):
            actions = sched.poll_and_rebalance()

        assert actions == {}
        mock_worker.steal_layers.remote.assert_not_called()

    def test_poll_and_rebalance_missing_gpu_skipped(self):
        """GPU not in poll result should be silently skipped."""
        sched = self._make_scheduler(gpu_ids=[0, 1], model_vram_gb=2.0)
        mock_worker = MagicMock()
        sched.set_workers({0: [mock_worker], 1: [mock_worker]})
        sched.set_work_queue(["L0"])

        # Only GPU 0 returned, GPU 1 missing
        with patch("octopus.scheduler.poll_gpu_memory", return_value={0: 0.5}):
            actions = sched.poll_and_rebalance()

        # GPU 0 below threshold, GPU 1 missing → no steal
        assert actions == {}

    def test_start_stop(self):
        """start() launches background thread; stop() joins it."""
        sched = self._make_scheduler()
        sched.set_workers({})
        sched.set_work_queue([])

        with patch("octopus.scheduler.poll_gpu_memory", return_value={}):
            sched.start()
            assert sched._running is True
            assert sched._thread is not None
            time.sleep(0.15)  # let it poll at least once
            sched.stop()
            assert sched._running is False
            assert sched._thread is None

    def test_poll_error_does_not_crash_thread(self):
        """Exceptions in poll_and_rebalance should be caught, not crash thread."""
        sched = self._make_scheduler()
        sched.set_workers({})
        sched.set_work_queue([])

        with patch("octopus.scheduler.poll_gpu_memory", side_effect=RuntimeError("boom")):
            sched.start()
            time.sleep(0.15)
            sched.stop()
        # If we get here, thread survived the exception

    def test_steal_distributes_across_workers(self):
        """Stolen layers should be distributed round-robin across workers on a GPU."""
        sched = self._make_scheduler(gpu_ids=[0], model_vram_gb=1.0)

        w0, w1 = MagicMock(), MagicMock()
        w0.steal_layers.remote.return_value = MagicMock()
        w1.steal_layers.remote.return_value = MagicMock()
        sched.set_workers({0: [w0, w1]})
        sched.set_work_queue(["L0", "L1", "L2"])

        # 10 GB free / 1.0 GB model = 10 extra slots, steal all 3
        with patch("octopus.scheduler.poll_gpu_memory", return_value={0: 10.0}):
            sched.poll_and_rebalance()

        total_calls = w0.steal_layers.remote.call_count + w1.steal_layers.remote.call_count
        assert total_calls == 3
