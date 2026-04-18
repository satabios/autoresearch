import pytest
from octopus._types import GPUInfo
from octopus.discovery import (
    compute_sharded_allocation,
    compute_worker_allocation,
)
from octopus.exceptions import InsufficientVRAMError


class TestComputeWorkerAllocation:
    def test_basic_allocation_2_gpus(self, mock_gpu_2x80):
        """Each GPU can fit multiple 10GB workers."""
        plan = compute_worker_allocation(mock_gpu_2x80, model_vram_gb=10.0, safety_net_gb=1.0)

        # GPU 0: (70 - 1) // 10 = 6 workers
        # GPU 1: (65 - 1) // 10 = 6 workers
        assert plan.total_workers == 12
        assert len(plan.allocations) == 2
        assert plan.allocations[0].num_workers == 6
        assert plan.allocations[1].num_workers == 6
        assert plan.sharding_required is False
        assert plan.gpus_per_shard == 1

    def test_basic_allocation_4_gpus(self, mock_gpu_4x24):
        """4 GPUs with 20GB available, 5GB model."""
        plan = compute_worker_allocation(mock_gpu_4x24, model_vram_gb=5.0, safety_net_gb=1.0)

        # Each GPU: (20 - 1) // 5 = 3 workers
        assert plan.total_workers == 12
        assert all(a.num_workers == 3 for a in plan.allocations)

    def test_large_safety_net(self, mock_gpu_2x80):
        """With 60GB safety net, very few workers fit."""
        plan = compute_worker_allocation(mock_gpu_2x80, model_vram_gb=5.0, safety_net_gb=60.0)

        # GPU 0: (70 - 60) // 5 = 2 workers
        # GPU 1: (65 - 60) // 5 = 1 worker
        assert plan.total_workers == 3

    def test_model_too_large_raises(self, mock_gpu_1x8):
        """Model larger than available VRAM should raise."""
        with pytest.raises(InsufficientVRAMError, match="sharding"):
            compute_worker_allocation(mock_gpu_1x8, model_vram_gb=10.0, safety_net_gb=1.0)

    def test_max_workers_cap(self, mock_gpu_2x80):
        """max_workers should cap total count."""
        plan = compute_worker_allocation(
            mock_gpu_2x80, model_vram_gb=10.0, safety_net_gb=1.0, max_workers=3
        )
        assert plan.total_workers == 3

    def test_workers_per_gpu_cap(self, mock_gpu_2x80):
        """workers_per_gpu=1 should keep one worker on each usable GPU."""
        plan = compute_worker_allocation(
            mock_gpu_2x80,
            model_vram_gb=10.0,
            safety_net_gb=1.0,
            workers_per_gpu=1,
        )
        assert plan.total_workers == 2
        assert all(alloc.num_workers == 1 for alloc in plan.allocations)

    def test_caps_workers_for_ray_fraction_precision(self):
        """Per-GPU worker count should never exceed Ray's 1e-4 GPU fraction limit."""
        gpus = [
            GPUInfo(
                device_id=0,
                name="Big",
                total_vram_gb=80.0,
                available_vram_gb=70.0,
                compute_capability=(8, 0),
            )
        ]
        plan = compute_worker_allocation(gpus, model_vram_gb=0.000001, safety_net_gb=0.0)
        assert plan.total_workers == 10_000
        assert plan.allocations[0].num_workers == 10_000

    def test_exact_fit(self):
        """Model exactly fills available VRAM after safety net."""
        gpus = [GPUInfo(device_id=0, name="G", total_vram_gb=16.0,
                        available_vram_gb=11.0, compute_capability=(8, 0))]
        plan = compute_worker_allocation(gpus, model_vram_gb=10.0, safety_net_gb=1.0)
        assert plan.total_workers == 1
        assert plan.allocations[0].num_workers == 1

    def test_zero_safety_net(self, mock_gpu_1x8):
        """With 0 safety net, we use all available VRAM."""
        plan = compute_worker_allocation(mock_gpu_1x8, model_vram_gb=3.0, safety_net_gb=0.0)
        # (6.0 - 0) // 3.0 = 2
        assert plan.total_workers == 2

    def test_heterogeneous_gpus(self):
        """GPUs with different VRAM amounts."""
        gpus = [
            GPUInfo(device_id=0, name="Big", total_vram_gb=80.0,
                    available_vram_gb=70.0, compute_capability=(8, 0)),
            GPUInfo(device_id=1, name="Small", total_vram_gb=16.0,
                    available_vram_gb=12.0, compute_capability=(8, 6)),
        ]
        plan = compute_worker_allocation(gpus, model_vram_gb=10.0, safety_net_gb=1.0)
        # GPU 0: (70-1)//10 = 6
        # GPU 1: (12-1)//10 = 1
        assert plan.total_workers == 7
        assert plan.allocations[0].num_workers == 6
        assert plan.allocations[1].num_workers == 1

    def test_gpu_with_insufficient_vram_skipped(self):
        """GPUs that can't fit even 1 worker should be skipped."""
        gpus = [
            GPUInfo(device_id=0, name="Big", total_vram_gb=80.0,
                    available_vram_gb=70.0, compute_capability=(8, 0)),
            GPUInfo(device_id=1, name="Tiny", total_vram_gb=4.0,
                    available_vram_gb=3.0, compute_capability=(7, 0)),
        ]
        plan = compute_worker_allocation(gpus, model_vram_gb=10.0, safety_net_gb=1.0)
        assert plan.total_workers == 6
        assert len(plan.allocations) == 1  # tiny GPU skipped


class TestComputeShardedAllocation:
    def test_basic_tp_sharding(self, mock_gpu_2x80):
        """Model needs 2 GPUs for TP."""
        plan = compute_sharded_allocation(
            mock_gpu_2x80, model_vram_gb=100.0, safety_net_gb=1.0, strategy="tp"
        )
        assert plan.sharding_required is True
        assert plan.sharding_strategy == "tp"
        # 100 * 1.10 = 110GB needed, each GPU has ~69GB usable
        # ceil(110 / 64) = 2 GPUs per shard (uses min available=64)
        assert plan.gpus_per_shard == 2
        assert plan.total_workers == 1  # 2 GPUs / 2 per shard = 1 group

    def test_pp_sharding(self, mock_gpu_4x24):
        """Model needs PP across 4 GPUs."""
        plan = compute_sharded_allocation(
            mock_gpu_4x24, model_vram_gb=60.0, safety_net_gb=1.0, strategy="pp"
        )
        assert plan.sharding_required is True
        assert plan.sharding_strategy == "pp"
        # 60 * 1.02 = 61.2GB, each GPU has 19GB usable
        # ceil(61.2 / 19) = 4 GPUs per shard
        assert plan.gpus_per_shard >= 3

    def test_insufficient_gpus_for_shard_raises(self, mock_gpu_1x8):
        """Not enough GPUs for sharding should raise."""
        with pytest.raises(InsufficientVRAMError):
            compute_sharded_allocation(
                mock_gpu_1x8, model_vram_gb=50.0, safety_net_gb=1.0, strategy="tp"
            )

    def test_multiple_shard_groups(self):
        """8 GPUs, model needs 2 GPUs -> 4 shard groups."""
        gpus = [
            GPUInfo(device_id=i, name="A100", total_vram_gb=80.0,
                    available_vram_gb=70.0, compute_capability=(8, 0))
            for i in range(8)
        ]
        plan = compute_sharded_allocation(
            gpus, model_vram_gb=100.0, safety_net_gb=1.0, strategy="tp"
        )
        # 100 * 1.10 = 110, each has 69 usable, ceil(110/69) = 2
        assert plan.gpus_per_shard == 2
        assert plan.total_workers == 4  # 8 / 2

    def test_shard_device_ids_preserved_for_non_contiguous_gpu_ids(self):
        """Shard groups should keep actual GPU ids, not contiguous ranges."""
        gpus = [
            GPUInfo(device_id=device_id, name="A100", total_vram_gb=80.0,
                    available_vram_gb=70.0, compute_capability=(8, 0))
            for device_id in (1, 3, 5, 7)
        ]
        plan = compute_sharded_allocation(
            gpus, model_vram_gb=100.0, safety_net_gb=1.0, strategy="tp"
        )
        assert plan.gpus_per_shard == 2
        assert plan.allocations[0].shard_device_ids == (1, 3)
        assert plan.allocations[1].shard_device_ids == (5, 7)

    def test_max_workers_cap_with_sharding(self):
        """max_workers should limit shard groups."""
        gpus = [
            GPUInfo(device_id=i, name="A100", total_vram_gb=80.0,
                    available_vram_gb=70.0, compute_capability=(8, 0))
            for i in range(8)
        ]
        plan = compute_sharded_allocation(
            gpus, model_vram_gb=100.0, safety_net_gb=1.0,
            strategy="tp", max_workers=2,
        )
        assert plan.total_workers == 2
