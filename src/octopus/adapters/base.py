from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol that all model adapters must satisfy."""

    @property
    def model_type_name(self) -> str: ...

    @property
    def eval_fn(self) -> Callable: ...

    def load_to_device(self, device: torch.device) -> None: ...

    def unload(self) -> None: ...

    def forward(self, batch: Any) -> Any: ...

    def state_bytes(self) -> bytes: ...

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn: Callable) -> ModelAdapter: ...
