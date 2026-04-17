import torch
import torch.nn as nn
import pytest

from octopus._types import GPUInfo, VRAMProfile, PoolPlan, WorkerAllocation


@pytest.fixture
def toy_model() -> nn.Module:
    """A tiny nn.Module for testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def toy_eval_fn():
    def eval_fn(model, batch):
        return model(batch)
    return eval_fn


@pytest.fixture
def mock_gpu_2x80() -> list[GPUInfo]:
    """Two mock A100 80GB GPUs."""
    return [
        GPUInfo(device_id=0, name="Mock A100", total_vram_gb=80.0,
                available_vram_gb=70.0, compute_capability=(8, 0)),
        GPUInfo(device_id=1, name="Mock A100", total_vram_gb=80.0,
                available_vram_gb=65.0, compute_capability=(8, 0)),
    ]


@pytest.fixture
def mock_gpu_4x24() -> list[GPUInfo]:
    """Four mock RTX 4090 24GB GPUs."""
    return [
        GPUInfo(device_id=i, name="Mock RTX 4090", total_vram_gb=24.0,
                available_vram_gb=20.0, compute_capability=(8, 9))
        for i in range(4)
    ]


@pytest.fixture
def mock_gpu_1x8() -> list[GPUInfo]:
    """One small GPU with limited VRAM."""
    return [
        GPUInfo(device_id=0, name="Mock RTX 3060", total_vram_gb=8.0,
                available_vram_gb=6.0, compute_capability=(8, 6)),
    ]
