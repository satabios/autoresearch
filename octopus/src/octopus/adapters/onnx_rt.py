from __future__ import annotations

import io
from typing import Any, Callable, Optional

from octopus._logging import get_logger
from octopus.adapters.base import ModelAdapter

_log = get_logger()


class ONNXRuntimeAdapter:
    """Adapter for onnxruntime.InferenceSession.

    Accepts either a live InferenceSession or a path to an .onnx file.
    Internally stores the model as raw bytes for shipping to Ray workers,
    and reconstructs a session per-worker with the appropriate CUDA EP.
    """

    def __init__(self, session_or_path: Any, eval_fn: Callable) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        self._eval_fn = eval_fn
        self._session: Optional[ort.InferenceSession] = None
        self._device_id: Optional[int] = None

        if isinstance(session_or_path, str):
            with open(session_or_path, "rb") as f:
                self._model_bytes = f.read()
        elif isinstance(session_or_path, ort.InferenceSession):
            # ORT sessions don't expose raw bytes directly;
            # we need the original model path or bytes
            model_path = session_or_path._model_path  # type: ignore[attr-defined]
            if model_path:
                with open(model_path, "rb") as f:
                    self._model_bytes = f.read()
            else:
                raise ValueError(
                    "Cannot extract model bytes from InferenceSession. "
                    "Pass the .onnx file path instead."
                )
        elif isinstance(session_or_path, bytes):
            self._model_bytes = session_or_path
        else:
            raise TypeError(f"Expected str path, ort.InferenceSession, or bytes, got {type(session_or_path)}")

    @property
    def model_type_name(self) -> str:
        return "onnx"

    @property
    def eval_fn(self) -> Callable:
        return self._eval_fn

    def load_to_device(self, device: Any) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        # Extract device_id from torch.device or int
        if hasattr(device, "index"):
            self._device_id = device.index if device.index is not None else 0
        else:
            self._device_id = int(device)

        providers = [
            ("CUDAExecutionProvider", {"device_id": self._device_id}),
            "CPUExecutionProvider",
        ]
        self._session = ort.InferenceSession(self._model_bytes, providers=providers)
        _log.info("ORT session loaded on device %d", self._device_id)

    def unload(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None

    def forward(self, batch: Any) -> Any:
        assert self._session is not None, "Session not loaded. Call load_to_device() first."
        return self._eval_fn(self._session, batch)

    def state_bytes(self) -> bytes:
        return self._model_bytes

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn: Callable) -> "ONNXRuntimeAdapter":
        return cls(data, eval_fn)
