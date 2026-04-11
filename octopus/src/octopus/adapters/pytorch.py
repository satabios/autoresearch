from __future__ import annotations

import copy
import io
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from octopus.adapters.base import ModelAdapter


class PyTorchAdapter:
    """Adapter for torch.nn.Module models."""

    def __init__(self, model: nn.Module, eval_fn: Callable[[Any, Any], Any]) -> None:
        self._original_model = model
        self._eval_fn = eval_fn
        self._device: Optional[torch.device] = None
        self._active_model: Optional[nn.Module] = None

    @property
    def model_type_name(self) -> str:
        return "pytorch"

    @property
    def eval_fn(self) -> Callable:
        return self._eval_fn

    def load_to_device(self, device: torch.device) -> None:
        self._device = device
        self._active_model = copy.deepcopy(self._original_model)
        self._active_model.to(device)
        self._active_model.eval()

    def unload(self) -> None:
        if self._active_model is not None:
            self._active_model.cpu()
            del self._active_model
            self._active_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, batch: Any) -> Any:
        assert self._active_model is not None, "Model not loaded. Call load_to_device() first."
        with torch.no_grad():
            batch_on_device = _to_device(batch, self._device)
            return self._eval_fn(self._active_model, batch_on_device)

    def state_bytes(self) -> bytes:
        buffer = io.BytesIO()
        # Save the full model (not just state_dict) for reliable reconstruction
        torch.save(self._original_model, buffer)
        return buffer.getvalue()

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn: Callable) -> "PyTorchAdapter":
        buffer = io.BytesIO(data)
        model = torch.load(buffer, map_location="cpu", weights_only=False)
        return cls(model, eval_fn)


def _to_device(obj: Any, device: Optional[torch.device]) -> Any:
    """Recursively move tensors to device."""
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj

