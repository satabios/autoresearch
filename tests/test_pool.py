from unittest.mock import MagicMock, patch

import pytest

from octopus._types import PoolPlan, WorkerAllocation


class TestWorkerPool:
    def test_start_creates_correct_worker_count(self):
        """WorkerPool.start() should create the right number of actors."""
        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=0, num_workers=3,
                                 vram_per_worker_gb=10.0, reserved_safety_gb=1.0),
                WorkerAllocation(device_id=1, num_workers=2,
                                 vram_per_worker_gb=10.0, reserved_safety_gb=1.0),
            ],
            total_workers=5,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.InferenceWorker") as MockWorker:

            # Create mock actor handle
            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "init_ref"

            # Mock the .options().remote() chain
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            MockWorker.options.return_value = mock_options

            mock_ray.get.return_value = [{"status": "ready"}] * 5

            from octopus.pool import WorkerPool
            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
            )
            pool.start()

            # Should create 5 actors total
            assert pool.num_workers == 5

    def test_submit_round_robins(self):
        """submit() should distribute batches round-robin."""
        from octopus.pool import WorkerPool

        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=0, num_workers=2,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
            ],
            total_workers=2,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        pool = WorkerPool(plan, b"fake", "pytorch", lambda m, b: None)

        # Manually inject mock workers
        w0, w1 = MagicMock(), MagicMock()
        w0.run.remote.return_value = "ref0"
        w1.run.remote.return_value = "ref1"
        pool._workers = [w0, w1]

        pool.submit("batch0")  # -> w0
        pool.submit("batch1")  # -> w1
        pool.submit("batch2")  # -> w0

        assert w0.run.remote.call_count == 2
        assert w1.run.remote.call_count == 1

    def test_shutdown(self):
        """shutdown() should call shutdown on all workers and clear list."""
        from octopus.pool import WorkerPool

        plan = PoolPlan(
            allocations=[], total_workers=0,
            sharding_required=False, sharding_strategy=None, gpus_per_shard=1,
        )

        with patch("octopus.pool.ray") as mock_ray:
            pool = WorkerPool(plan, b"", "pytorch", lambda m, b: None)
            w0 = MagicMock()
            pool._workers = [w0]

            pool.shutdown()

            assert pool.num_workers == 0
            w0.shutdown.remote.assert_called_once()

    def test_start_sharded_uses_planned_group_gpu_ids(self):
        """Sharded startup should pass recorded shard_device_ids to actor."""
        plan = PoolPlan(
            allocations=[
                WorkerAllocation(
                    device_id=2,
                    num_workers=1,
                    vram_per_worker_gb=10.0,
                    reserved_safety_gb=1.0,
                    shard_device_ids=(2, 5),
                ),
            ],
            total_workers=1,
            sharding_required=True,
            sharding_strategy="pp",
            gpus_per_shard=2,
        )

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.ShardedInferenceWorkerGroup") as MockWorker:

            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "init_ref"
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            MockWorker.options.return_value = mock_options
            mock_ray.get.return_value = [{"status": "ready"}]

            from octopus.pool import WorkerPool

            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
            )
            pool.start()

            MockWorker.options.assert_called_once_with(num_gpus=2)
            mock_options.remote.assert_called_once()
            assert mock_options.remote.call_args.args[3] == [2, 5]
