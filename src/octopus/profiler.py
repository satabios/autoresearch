from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any, Callable

import torch

from octopus._logging import get_logger
from octopus._types import GPUInfo, VRAMProfile
from octopus.adapters.base import ModelAdapter
from octopus.exceptions import ProfilingError

_log = get_logger()

_VRAM_PROBE_SAFETY_MULTIPLIER = 1.25  # QuantSim overhead vs plain ORT
_VRAM_PROBE_FALLBACK_GB = 4.0


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


def profile_ort_vram(
    onnx_path: str,
    sample_feed_fn: Callable,
    gpu_id: int,
    n_warmup: int = 20,
) -> VRAMProfile:
    """Subprocess VRAM probe for ONNX models using nvidia-smi deltas.

    Runs in an isolated subprocess to avoid polluting the parent's CUDA context
    (an ORT CUDA session creates a persistent context of ~200-400 MB even after
    the session is deleted).

    Uses nvidia-smi with the physical GPU ID (ignores CUDA_VISIBLE_DEVICES).
    Applies a 1.25x multiplier to account for QuantSim overhead vs plain ORT.
    Falls back to 4.0 GB with a warning if the probe fails.

    Args:
        onnx_path: Path to the .onnx model file.
        sample_feed_fn: Callable () -> dict[str, np.ndarray] producing one feed dict.
        gpu_id: Physical GPU index (used for both CUDA_VISIBLE_DEVICES and nvidia-smi).
        n_warmup: Number of forward passes to run for peak measurement.

    Returns:
        VRAMProfile with peak_vram_gb including the safety multiplier.
    """
    # Serialize sample_feed_fn via cloudpickle so the subprocess can call it.
    try:
        import cloudpickle  # type: ignore[import-untyped]
        feed_bytes_hex = cloudpickle.dumps(sample_feed_fn).hex()
    except ImportError:
        _log.warning(
            "cloudpickle not installed — profile_ort_vram falling back to %.1f GB.",
            _VRAM_PROBE_FALLBACK_GB,
        )
        return VRAMProfile(
            peak_vram_bytes=int(_VRAM_PROBE_FALLBACK_GB * (1 << 30)),
            peak_vram_gb=_VRAM_PROBE_FALLBACK_GB,
            model_params_bytes=0,
            activation_peak_bytes=0,
            profiled_on_device=gpu_id,
        )

    probe_script = textwrap.dedent(f"""
import os, sys, json, subprocess, time
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import cloudpickle, numpy as np
import onnxruntime as ort

def _nvml_free_mb(physical_gpu_id):
    out = subprocess.check_output([
        "nvidia-smi",
        "--id=" + str(physical_gpu_id),
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    return float(out.split("\\n")[0].strip())

feed_fn = cloudpickle.loads(bytes.fromhex("{feed_bytes_hex}"))
onnx_path = {onnx_path!r}
n_warmup = {n_warmup}
physical_gpu_id = {gpu_id}

baseline_mb = _nvml_free_mb(physical_gpu_id)

providers = [("CUDAExecutionProvider", {{"device_id": 0}}), "CPUExecutionProvider"]
sess = ort.InferenceSession(onnx_path, providers=providers)

# Verify CUDAExecutionProvider is active (not silent CPU fallback)
active = [p for p in sess.get_providers() if "CUDA" in p]
if not active:
    print(json.dumps({{"error": "CUDAExecutionProvider not active"}}))
    sys.exit(1)

after_load_mb = _nvml_free_mb(physical_gpu_id)
peak_mb = after_load_mb

for _ in range(n_warmup):
    feed = feed_fn()
    sess.run(None, feed)
    cur_mb = _nvml_free_mb(physical_gpu_id)
    if cur_mb < peak_mb:
        peak_mb = cur_mb

# delta = memory consumed (baseline was higher = more free)
delta_mb = baseline_mb - peak_mb
print(json.dumps({{"baseline_mb": baseline_mb, "after_load_mb": after_load_mb, "peak_mb": peak_mb, "delta_mb": delta_mb}}))
""")

    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Probe subprocess failed: {result.stderr[-500:]}")

        # Last line of stdout is the JSON payload
        last_line = result.stdout.strip().split("\n")[-1]
        data = json.loads(last_line)

        if "error" in data:
            raise RuntimeError(data["error"])

        delta_gb = data["delta_mb"] / 1024.0 * _VRAM_PROBE_SAFETY_MULTIPLIER
        delta_gb = max(delta_gb, 0.1)  # floor at 100 MB

        _log.info(
            "ORT VRAM probe on GPU %d: baseline=%.0f MB, peak=%.0f MB, "
            "delta=%.2f GB (×%.2f → %.2f GB)",
            gpu_id,
            data["baseline_mb"],
            data["peak_mb"],
            data["delta_mb"] / 1024.0,
            _VRAM_PROBE_SAFETY_MULTIPLIER,
            delta_gb,
        )

        return VRAMProfile(
            peak_vram_bytes=int(delta_gb * (1 << 30)),
            peak_vram_gb=delta_gb,
            model_params_bytes=int((data["baseline_mb"] - data["after_load_mb"]) / 1024.0 * (1 << 30)),
            activation_peak_bytes=0,
            profiled_on_device=gpu_id,
        )

    except Exception as e:
        _log.warning(
            "profile_ort_vram probe failed (%s) — falling back to %.1f GB.",
            e,
            _VRAM_PROBE_FALLBACK_GB,
        )
        return VRAMProfile(
            peak_vram_bytes=int(_VRAM_PROBE_FALLBACK_GB * (1 << 30)),
            peak_vram_gb=_VRAM_PROBE_FALLBACK_GB,
            model_params_bytes=0,
            activation_peak_bytes=0,
            profiled_on_device=gpu_id,
        )
