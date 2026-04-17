import io
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from octopus._types import VRAMProfile


class TestProfilerUnit:
    """Unit tests for profiler with mocked CUDA calls."""

    def test_profile_returns_vram_profile(self):
        """profile_model_vram returns VRAMProfile with correct fields."""
        from octopus.profiler import profile_model_vram

        # Create a real model adapter with a tiny model
        model = nn.Linear(10, 5)
        from octopus.adapters.pytorch import PyTorchAdapter
        eval_fn = lambda m, b: m(b)
        adapter = PyTorchAdapter(model, eval_fn)

        sample_batch = torch.randn(2, 10)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.memory_allocated", side_effect=[
                 0,            # baseline
                 1000,         # after_load
             ]), \
             patch("torch.cuda.max_memory_allocated", return_value=5000), \
             patch("torch.cuda.synchronize"), \
             patch("torch.cuda.empty_cache"), \
             patch.object(adapter, "load_to_device"), \
             patch.object(adapter, "forward"), \
             patch.object(adapter, "unload"):

            profile = profile_model_vram(adapter, sample_batch, device_id=0)

        assert isinstance(profile, VRAMProfile)
        assert profile.peak_vram_bytes == 5000
        assert profile.model_params_bytes == 1000
        assert profile.activation_peak_bytes == 4000
        assert profile.profiled_on_device == 0

    def test_pick_profiling_gpu_selects_most_free(self):
        """pick_profiling_gpu should select GPU with most available VRAM."""
        from octopus._types import GPUInfo
        from octopus.profiler import pick_profiling_gpu

        gpus = [
            GPUInfo(0, "G0", 80.0, 40.0, (8, 0)),
            GPUInfo(1, "G1", 80.0, 70.0, (8, 0)),
            GPUInfo(2, "G2", 80.0, 55.0, (8, 0)),
        ]
        assert pick_profiling_gpu(gpus) == 1
