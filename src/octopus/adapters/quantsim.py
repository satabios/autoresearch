from __future__ import annotations

import io
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from octopus._logging import get_logger
from octopus.adapters.base import ModelAdapter
from octopus.adapters.pytorch import _to_device

_log = get_logger()


class QuantSimAdapter:
    """Adapter for AIMET QuantizationSimModel.

    QuantSim wraps a PyTorch model with quantization-aware wrappers.
    This adapter preserves both the model weights and quantization
    encodings (scale, offset, bitwidth) when serializing to workers.
    """

    def __init__(self, quantsim_model: Any, eval_fn: Callable) -> None:
        self._qsim = quantsim_model
        self._eval_fn = eval_fn
        self._device: Optional[torch.device] = None

    @property
    def model_type_name(self) -> str:
        return "quantsim"

    @property
    def eval_fn(self) -> Callable:
        return self._eval_fn

    def load_to_device(self, device: torch.device) -> None:
        self._device = device
        self._qsim.model.to(device)
        self._qsim.model.eval()

    def unload(self) -> None:
        if self._qsim is not None and hasattr(self._qsim, "model"):
            self._qsim.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, batch: Any) -> Any:
        assert self._device is not None, "Model not loaded. Call load_to_device() first."
        with torch.no_grad():
            batch_on_device = _to_device(batch, self._device)
            return self._eval_fn(self._qsim.model, batch_on_device)

    def state_bytes(self) -> bytes:
        buffer = io.BytesIO()
        # Serialize state dict and quantization encodings
        save_data: dict[str, Any] = {
            "state_dict": self._qsim.model.state_dict(),
        }
        # Try to export quantization encodings
        try:
            # AIMET provides export method for encodings
            import tempfile
            import json
            import os

            with tempfile.TemporaryDirectory() as tmpdir:
                enc_path = os.path.join(tmpdir, "encodings")
                self._qsim.export(tmpdir, "encodings")
                # Read back the encoding file
                enc_file = enc_path + ".encodings"
                if os.path.exists(enc_file):
                    with open(enc_file, "r") as f:
                        save_data["encodings"] = json.load(f)
        except Exception:
            _log.warning("Could not export quantization encodings; only state_dict will be serialized.")

        # Store the model class info for reconstruction
        save_data["model_cls"] = type(self._qsim.model)
        torch.save(save_data, buffer)
        return buffer.getvalue()

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn: Callable) -> "QuantSimAdapter":
        """Reconstruct QuantSim adapter from serialized bytes.

        Note: full round-trip requires AIMET to be installed in the worker.
        Falls back to a standard PyTorch model if QuantSim reconstruction fails.
        """
        buffer = io.BytesIO(data)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)

        model_cls = checkpoint["model_cls"]
        try:
            model = model_cls()
        except TypeError:
            model = model_cls.__new__(model_cls)
            if hasattr(model, "__init__"):
                try:
                    model.__init__()
                except TypeError:
                    pass

        model.load_state_dict(checkpoint["state_dict"])

        try:
            from aimet_torch.quantsim import QuantizationSimModel  # type: ignore[import-untyped]

            # Re-create QuantSim wrapper
            qsim = QuantizationSimModel(model, dummy_input=torch.randn(1))

            # Restore encodings if available
            if "encodings" in checkpoint:
                import tempfile
                import json
                import os

                with tempfile.TemporaryDirectory() as tmpdir:
                    enc_path = os.path.join(tmpdir, "encodings.encodings")
                    with open(enc_path, "w") as f:
                        json.dump(checkpoint["encodings"], f)
                    qsim.load_encodings(enc_path)

            return cls(qsim, eval_fn)
        except ImportError:
            _log.warning(
                "AIMET not available in worker; using model without quantization wrappers."
            )
            # Fallback: create a minimal mock that acts like QuantSim
            mock_qsim = type("MockQuantSim", (), {"model": model})()
            return cls(mock_qsim, eval_fn)
