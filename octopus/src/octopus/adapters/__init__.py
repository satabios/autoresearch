from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from octopus.adapters.base import ModelAdapter
from octopus.adapters.pytorch import PyTorchAdapter


def detect_and_wrap(model: Any, eval_fn: Callable) -> ModelAdapter:
    """Auto-detect model type and return the appropriate adapter.

    Detection order:
        1. AIMET QuantizationSimModel (wraps nn.Module, must check first)
        2. ONNX Runtime InferenceSession
        3. torch.nn.Module (catch-all for PyTorch)

    Raises:
        TypeError: if model type is not recognized.
    """
    # AIMET QuantSim check
    try:
        from octopus.adapters.quantsim import QuantSimAdapter

        from aimet_torch.quantsim import QuantizationSimModel  # type: ignore[import-untyped]

        if isinstance(model, QuantizationSimModel):
            return QuantSimAdapter(model, eval_fn)
    except ImportError:
        pass

    # ONNX Runtime check
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        from octopus.adapters.onnx_rt import ONNXRuntimeAdapter

        if isinstance(model, ort.InferenceSession):
            return ONNXRuntimeAdapter(model, eval_fn)
    except ImportError:
        pass

    # PyTorch check (most general, last)
    if isinstance(model, nn.Module):
        return PyTorchAdapter(model, eval_fn)

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected nn.Module, ort.InferenceSession, or QuantizationSimModel."
    )


# Lookup for reconstructing adapters in workers from type name
_ADAPTER_REGISTRY: dict[str, type] = {
    "pytorch": PyTorchAdapter,
}


def _register_lazy() -> None:
    """Register optional adapters if their dependencies are importable."""
    try:
        from octopus.adapters.onnx_rt import ONNXRuntimeAdapter

        _ADAPTER_REGISTRY["onnx"] = ONNXRuntimeAdapter
    except ImportError:
        pass
    try:
        from octopus.adapters.quantsim import QuantSimAdapter

        _ADAPTER_REGISTRY["quantsim"] = QuantSimAdapter
    except ImportError:
        pass


def resolve_adapter_class(name: str) -> type:
    """Resolve adapter class by model_type_name string."""
    if name not in _ADAPTER_REGISTRY:
        _register_lazy()
    if name not in _ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter type: {name}")
    return _ADAPTER_REGISTRY[name]
