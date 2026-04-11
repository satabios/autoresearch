from __future__ import annotations

from typing import Any

import torch

from octopus._logging import get_logger
from octopus._types import GPUInfo, VRAMProfile
from octopus.adapters.base import ModelAdapter
from octopus.exceptions import ProfilingError

_log = get_logger()


def profile_model_vram(
    adapter: ModelAdapter,
    sample_batch: Any,
    device_id: int = 0,
) -> VRAMProfile:
    """Load model onto one GPU, run one forward pass, measure peak VRAM.

    Args:
        adapter: wrapped model adapter.
        sample_batch: one representative batch for forward pass.
        device_id: CUDA ordinal to profile on.

    Returns:
        VRAMProfile with measured VRAM usage.

    Raises:
        ProfilingError: on OOM or forward pass failure.
    """
    device = torch.device(f"cuda:{device_id}")
    try:
        torch.cuda.reset_peak_memory_stats(device_id)
        baseline = torch.cuda.memory_allocated(device_id)

        adapter.load_to_device(device)
        after_load = torch.cuda.memory_allocated(device_id)
        model_params_bytes = after_load - baseline

        adapter.forward(sample_batch)
        torch.cuda.synchronize(device_id)

        peak = torch.cuda.max_memory_allocated(device_id)
        peak_usage = peak - baseline
        activation_bytes = peak_usage - model_params_bytes

        _log.info(
            "VRAM profile on cuda:%d: peak=%.2f GB (params=%.2f GB, activations=%.2f GB)",
            device_id,
            peak_usage / (1 << 30),
            model_params_bytes / (1 << 30),
            activation_bytes / (1 << 30),
        )

        return VRAMProfile(
            peak_vram_bytes=peak_usage,
            peak_vram_gb=peak_usage / (1 << 30),
            model_params_bytes=model_params_bytes,
            activation_peak_bytes=activation_bytes,
            profiled_on_device=device_id,
        )
    except torch.cuda.OutOfMemoryError as e:
        raise ProfilingError(
            f"OOM during profiling on cuda:{device_id}. "
            f"Try profiling on a GPU with more free VRAM."
        ) from e
    except Exception as e:
        raise ProfilingError(f"Profiling failed: {e}") from e
    finally:
        adapter.unload()
        torch.cuda.empty_cache()


def pick_profiling_gpu(gpus: list[GPUInfo]) -> int:
    """Select GPU with most available VRAM for profiling."""
    return max(gpus, key=lambda g: g.available_vram_gb).device_id
