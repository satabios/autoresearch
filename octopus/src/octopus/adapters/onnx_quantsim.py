from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from typing import Any, Callable, Optional

from octopus._logging import get_logger

_log = get_logger()

# Maps qscheme_key → (param_type, activation_type) for aimet_onnx
# Populated lazily to avoid hard import at module load time.
_QSCHEME_TYPES: Optional[dict] = None


def _get_qscheme_types() -> dict:
    global _QSCHEME_TYPES
    if _QSCHEME_TYPES is not None:
        return _QSCHEME_TYPES
    from aimet_common.defs import QuantizationDataType  # type: ignore[import-untyped]
    from aimet_onnx.quantsim import QuantizationSimModel  # type: ignore[import-untyped]  # noqa: F401

    # aimet_onnx uses QuantizationDataType.int for integer quantization
    int_t = QuantizationDataType.int
    float_t = QuantizationDataType.float
    _QSCHEME_TYPES = {
        "w8a8": (int_t, int_t),
        "w8a16": (int_t, int_t),   # activation bitwidth set via config
        "w4a8": (int_t, int_t),
        "w4a16": (int_t, int_t),
        "fp16": (float_t, float_t),
    }
    return _QSCHEME_TYPES


class OnnxQuantSimAdapter:
    """Adapter for AIMET ONNX QuantizationSimModel.

    Serialization path:
        state_bytes()      → sim.export() → tar(.onnx + .encodings + metadata.json)
        from_state_bytes() → untar → onnx.load() → QuantizationSimModel()
                             → load_encodings_to_sim(strict=False)
    """

    def __init__(
        self,
        sim: Any,
        eval_fn: Callable,
        qscheme_key: str = "w8a8",
        config_file: str = "htp_v81",
    ) -> None:
        self._sim = sim
        self._eval_fn = eval_fn
        self._qscheme_key = qscheme_key
        self._config_file = config_file

    @property
    def model_type_name(self) -> str:
        return "onnx_quantsim"

    @property
    def eval_fn(self) -> Callable:
        return self._eval_fn

    def load_to_device(self, device: Any) -> None:
        # ORT handles device placement at QuantSim construction via providers.
        pass

    def unload(self) -> None:
        """Delete ORT session to free GPU memory."""
        if hasattr(self._sim, "session") and self._sim.session is not None:
            del self._sim.session
            self._sim.session = None

    def forward(self, batch: Any) -> Any:
        return self._eval_fn(self._sim.session)

    def state_bytes(self) -> bytes:
        """Export sim to tar containing .onnx + .encodings + metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._sim.export(path=tmpdir, filename_prefix="worker_sim", export_model=True)

            # Locate .onnx file
            onnx_path = os.path.join(tmpdir, "worker_sim.onnx")
            if not os.path.exists(onnx_path):
                raise RuntimeError(f"sim.export() did not produce worker_sim.onnx in {tmpdir}")

            # Locate encodings — AIMET writes either .encodings or .encodings.json
            enc_path = os.path.join(tmpdir, "worker_sim.encodings")
            if not os.path.exists(enc_path):
                enc_path = os.path.join(tmpdir, "worker_sim.encodings.json")
            if not os.path.exists(enc_path):
                raise RuntimeError(
                    f"sim.export() did not produce .encodings or .encodings.json in {tmpdir}"
                )

            # Write metadata
            meta = {"qscheme_key": self._qscheme_key, "config_file": self._config_file}
            meta_path = os.path.join(tmpdir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            # Tar all three files
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(onnx_path, arcname="worker_sim.onnx")
                tar.add(enc_path, arcname="worker_sim.encodings")
                tar.add(meta_path, arcname="metadata.json")
            return buf.getvalue()

    @classmethod
    def from_state_bytes(cls, data: bytes, eval_fn: Callable) -> "OnnxQuantSimAdapter":
        """Reconstruct adapter on worker side from tarball bytes."""
        from aimet_common.defs import QuantScheme  # type: ignore[import-untyped]
        from aimet_onnx.quantsim import QuantizationSimModel  # type: ignore[import-untyped]
        from aimet_onnx.utils import load_encodings_to_sim  # type: ignore[import-untyped]
        import onnx  # type: ignore[import-untyped]

        buf = io.BytesIO(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                tar.extractall(tmpdir, filter="data")

            onnx_path = os.path.join(tmpdir, "worker_sim.onnx")
            enc_path = os.path.join(tmpdir, "worker_sim.encodings")
            meta_path = os.path.join(tmpdir, "metadata.json")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            qscheme_key = meta["qscheme_key"]
            config_file = meta["config_file"]

            qscheme_types = _get_qscheme_types()
            param_type, activation_type = qscheme_types.get(
                qscheme_key, qscheme_types["w8a8"]
            )

            onnx_model = onnx.load(onnx_path)

            # Workers always see device 0 via CUDA_VISIBLE_DEVICES
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]

            sim = QuantizationSimModel(
                model=onnx_model,
                quant_scheme=QuantScheme.post_training_tf,
                config_file=config_file,
                param_type=param_type,
                activation_type=activation_type,
                providers=providers,
            )

            # strict=False: allows bitwidth mismatches between fresh QuantSim
            # and exported encodings from a mixed-precision state.
            load_encodings_to_sim(sim, enc_path, strict=False)

            return cls(sim, eval_fn, qscheme_key=qscheme_key, config_file=config_file)
