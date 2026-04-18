from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Callable, Optional

from aimet_onnx.quantsim import QuantizationSimModel as _Original  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)

_MODE_PASSTHROUGH = "passthrough"
_MODE_REPLICATED = "replicated"
_MODE_SHARDED = "sharded"


def _make_zero_feed_fn(model_proto_bytes: bytes) -> Callable:
    """Return a callable () -> dict[str, np.ndarray] with zero inputs matching the model."""

    def _feed() -> dict:
        import numpy as np  # type: ignore[import-untyped]
        import onnx  # type: ignore[import-untyped]

        model = onnx.ModelProto()
        model.ParseFromString(model_proto_bytes)

        _DTYPE_MAP = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            5: np.int32,
            6: np.int64,
            9: np.bool_,
            10: np.float16,
            11: np.float64,
        }
        feed: dict = {}
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else 1)
            dtype = _DTYPE_MAP.get(inp.type.tensor_type.elem_type, np.float32)
            feed[inp.name] = np.zeros(shape, dtype=dtype)
        return feed

    return _feed


def _is_histogram_scheme(sim: Any) -> bool:
    """Return True if sim uses a histogram-based calibration scheme.

    Histogram schemes (percentile, KL, tf_enhanced) require global statistics
    across all batches. Splitting data across workers and merging min/max gives
    a conservative (wider) range — so these must use single-worker passthrough.
    """
    try:
        try:
            from aimet_onnx.common.defs import QuantScheme  # type: ignore[import-untyped]
        except ImportError:
            from aimet_common.defs import QuantScheme  # type: ignore[import-untyped]

        return sim.quant_scheme in (
            QuantScheme.post_training_tf_enhanced,
            QuantScheme.post_training_percentile,
        )
    except Exception:
        return False


def _detect_mode(sim: "OctopusQuantSimModel") -> str:
    """Profile model VRAM and classify as sharded, replicated, or passthrough."""
    if _is_histogram_scheme(sim):
        _log.info(
            "Mode: passthrough (histogram-based quant scheme requires global statistics)"
        )
        return _MODE_PASSTHROUGH

    try:
        from octopus.discovery import discover_gpus
        from octopus.profiler import profile_ort_vram
    except ImportError:
        _log.debug("octopus not importable — using passthrough mode")
        return _MODE_PASSTHROUGH

    try:
        gpus = discover_gpus()
    except Exception as e:
        _log.debug("GPU discovery failed: %s", e)
        return _MODE_PASSTHROUGH

    if not gpus:
        return _MODE_PASSTHROUGH

    max_free_gb = max(g.available_vram_gb for g in gpus)
    profiling_gpu = max(gpus, key=lambda g: g.available_vram_gb).device_id

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "_probe.onnx")
        try:
            with open(onnx_path, "wb") as f:
                f.write(sim._octopus_model_proto_bytes)
        except Exception as e:
            _log.debug("Failed to write probe model: %s", e)
            return _MODE_PASSTHROUGH

        try:
            sample_feed_fn = _make_zero_feed_fn(sim._octopus_model_proto_bytes)
            profile = profile_ort_vram(
                onnx_path=onnx_path,
                sample_feed_fn=sample_feed_fn,
                gpu_id=profiling_gpu,
            )
            vram_needed_gb = profile.peak_vram_gb
        except Exception as e:
            _log.debug("VRAM profiling failed, using passthrough: %s", e)
            return _MODE_PASSTHROUGH

    _log.info(
        "Mode detection: model=%.2f GB, max_free=%.2f GB",
        vram_needed_gb,
        max_free_gb,
    )
    if vram_needed_gb > max_free_gb * 0.9:
        _log.info("Mode: sharded (model exceeds 90%% of largest GPU free VRAM)")
        return _MODE_SHARDED
    _log.info("Mode: replicated (model fits on GPU)")
    return _MODE_REPLICATED


class OctopusQuantSimModel(_Original):
    """Drop-in replacement for aimet_onnx.QuantizationSimModel.

    Intercepts compute_encodings() to dispatch to distributed calibration:
      - 'replicated': model fits on GPU → K parallel workers, data split K ways,
        encodings merged via per-tensor min/max.
      - 'sharded': model too large for one GPU → sequential stage calibration
        with activation buffering via session instrumentation.
      - 'passthrough': no GPUs or profiling failed → delegate to super().

    Mode is determined lazily on first compute_encodings() call to avoid
    blocking I/O during object construction.
    """

    def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
        # Serialize original proto BEFORE super().__init__ adds QDQ nodes.
        # Workers need the un-quantized ONNX to reconstruct fresh QuantSim.
        try:
            self._octopus_model_proto_bytes: bytes = model.SerializeToString()
        except AttributeError:
            # model may be a path string in some AIMET versions; read it
            try:
                with open(model, "rb") as f:
                    self._octopus_model_proto_bytes = f.read()
            except (TypeError, OSError):
                self._octopus_model_proto_bytes = b""

        # Capture init kwargs for worker QuantSim reconstruction.
        # 'providers' will be overridden per-worker (device_id=0 in CUDA_VISIBLE_DEVICES scope).
        self._octopus_init_kwargs: dict[str, Any] = dict(kwargs)
        self._octopus_mode: Optional[str] = None  # lazy

        super().__init__(model, *args, **kwargs)

    def _get_mode(self) -> str:
        if self._octopus_mode is None:
            self._octopus_mode = _detect_mode(self)
        return self._octopus_mode

    def compute_encodings(
        self,
        forward_pass_callback: Callable,
        forward_pass_callback_args: Any,
    ) -> None:
        mode = self._get_mode()
        if mode == _MODE_REPLICATED:
            from octopus_aimet._calibration import parallel_compute_encodings

            parallel_compute_encodings(self, forward_pass_callback, forward_pass_callback_args)
        elif mode == _MODE_SHARDED:
            from octopus_aimet._calibration import sharded_compute_encodings

            sharded_compute_encodings(self, forward_pass_callback, forward_pass_callback_args)
        else:
            super().compute_encodings(forward_pass_callback, forward_pass_callback_args)

    def sensitivity_scan(
        self,
        eval_fn: Callable,
        layers: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Distributed per-layer SQNR sensitivity scan.

        Args:
            eval_fn: Callable(ort.InferenceSession) -> float (e.g. returns SQNR or accuracy).
            layers: op names to evaluate. None = all quantized ops.

        Returns:
            {layer_name: sqnr_score}
        """
        from octopus_aimet._sensitivity import run_sensitivity_scan

        if layers is None:
            layers = list(getattr(self, "qc_quantize_op_dict", {}).keys())
        return run_sensitivity_scan(self, eval_fn, layers)
