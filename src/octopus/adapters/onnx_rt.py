from __future__ import annotations

import io
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from octopus._logging import get_logger
from octopus.adapters.base import ModelAdapter

_log = get_logger()

_BUNDLE_MODEL_NAME = "model.onnx"


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
        self._bundle_bytes: bytes
        self._bundle_dir: Optional[tempfile.TemporaryDirectory[str]] = None

        if isinstance(session_or_path, str):
            self._bundle_bytes = _bundle_model_artifacts(Path(session_or_path))
        elif isinstance(session_or_path, ort.InferenceSession):
            # ORT sessions don't expose raw bytes directly;
            # we need the original model path or bytes
            model_path = session_or_path._model_path  # type: ignore[attr-defined]
            if model_path:
                self._bundle_bytes = _bundle_model_artifacts(Path(model_path))
            else:
                raise ValueError(
                    "Cannot extract model bytes from InferenceSession. "
                    "Pass the .onnx file path instead."
                )
        elif isinstance(session_or_path, bytes):
            if _is_model_bundle(session_or_path):
                self._bundle_bytes = session_or_path
            else:
                self._bundle_bytes = _bundle_inline_model(session_or_path)
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
            (
                "CUDAExecutionProvider",
                {"device_id": self._device_id, "use_tf32": 0},
            ),
            "CPUExecutionProvider",
        ]
        model_path = _extract_model_bundle(self._bundle_bytes, self)
        self._session = ort.InferenceSession(model_path, providers=providers)
        _log.info("ORT session loaded on device %d", self._device_id)

    def unload(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None
        if self._bundle_dir is not None:
            self._bundle_dir.cleanup()
            self._bundle_dir = None

    def forward(self, batch: Any) -> Any:
        assert self._session is not None, "Session not loaded. Call load_to_device() first."
        return self._eval_fn(self._session, batch)

    def state_bytes(self) -> bytes:
        return self._bundle_bytes

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn: Callable) -> "ONNXRuntimeAdapter":
        return cls(data, eval_fn)


def _bundle_inline_model(model_bytes: bytes) -> bytes:
    """Wrap raw ONNX bytes in tarball so workers can extract to a temp path."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(_BUNDLE_MODEL_NAME)
        info.size = len(model_bytes)
        tar.addfile(info, io.BytesIO(model_bytes))
    return buffer.getvalue()


def _bundle_model_artifacts(model_path: Path) -> bytes:
    """Bundle ONNX model plus any external-data sidecars into one tarball."""
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model path does not exist: {model_path}")

    siblings = [model_path]
    prefix = model_path.name
    for candidate in sorted(model_path.parent.iterdir()):
        if candidate == model_path or not candidate.is_file():
            continue
        if candidate.name.startswith(f"{prefix}."):
            siblings.append(candidate)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for artifact in siblings:
            tar.add(artifact, arcname=artifact.name)
    return buffer.getvalue()


def _extract_model_bundle(bundle_bytes: bytes, adapter: ONNXRuntimeAdapter) -> str:
    """Extract bundled model artifacts to temp dir and return main model path."""
    if adapter._bundle_dir is not None:
        adapter._bundle_dir.cleanup()
    adapter._bundle_dir, model_path = extract_model_bundle_to_tempdir(bundle_bytes)
    return model_path


def _write_legacy_raw_model(bundle_root: Path, model_bytes: bytes) -> str:
    """Backward-compat path for old state_bytes() payloads that were raw model bytes."""
    model_path = bundle_root / _BUNDLE_MODEL_NAME
    model_path.write_bytes(model_bytes)
    return os.fspath(model_path)


def _is_model_bundle(payload: bytes) -> bool:
    """Return True when payload is already one of our tar.gz model bundles."""
    buffer = io.BytesIO(payload)
    try:
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            return any(name.endswith(".onnx") for name in tar.getnames())
    except tarfile.ReadError:
        return False


def extract_model_bundle_to_tempdir(
    bundle_bytes: bytes,
) -> tuple[tempfile.TemporaryDirectory[str], str]:
    """Extract model bundle bytes to temp dir and return (tmpdir, .onnx path)."""
    temp_dir = tempfile.TemporaryDirectory()
    bundle_root = Path(temp_dir.name)
    buffer = io.BytesIO(bundle_bytes)
    try:
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(bundle_root, filter="data")
            model_members = [
                member.name
                for member in tar.getmembers()
                if member.isfile() and member.name.endswith(".onnx")
            ]
    except tarfile.ReadError:
        return temp_dir, _write_legacy_raw_model(bundle_root, bundle_bytes)

    if not model_members:
        return temp_dir, _write_legacy_raw_model(bundle_root, bundle_bytes)
    model_path = bundle_root / model_members[0]
    return temp_dir, os.fspath(model_path)
