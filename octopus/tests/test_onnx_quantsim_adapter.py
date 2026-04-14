"""Tests for OnnxQuantSimAdapter (mocked — no AIMET/ONNX required)."""
from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter, _get_qscheme_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tarball(onnx_bytes=b"fake_onnx", enc_bytes=b"fake_enc", meta=None):
    """Build a minimal tar.gz matching what state_bytes() produces."""
    if meta is None:
        meta = {"qscheme_key": "w8a8", "config_file": "htp_v81"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [
            ("worker_sim.onnx", onnx_bytes),
            ("worker_sim.encodings", enc_bytes),
            ("metadata.json", json.dumps(meta).encode()),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _mock_sim(has_session=True):
    sim = MagicMock()
    if has_session:
        sim.session = MagicMock()
    else:
        del sim.session  # no session attribute
    return sim


# ---------------------------------------------------------------------------
# model_type_name
# ---------------------------------------------------------------------------

class TestModelTypeName:
    def test_returns_onnx_quantsim(self):
        sim = _mock_sim()
        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        assert adapter.model_type_name == "onnx_quantsim"


# ---------------------------------------------------------------------------
# load_to_device / unload
# ---------------------------------------------------------------------------

class TestLoadUnload:
    def test_load_to_device_is_noop(self):
        sim = _mock_sim()
        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        adapter.load_to_device("cuda:0")  # must not raise

    def test_unload_deletes_session(self):
        sim = _mock_sim(has_session=True)
        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        adapter.unload()
        assert sim.session is None

    def test_unload_no_session_attr_is_safe(self):
        sim = MagicMock(spec=[])  # no attributes
        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        adapter.unload()  # must not raise


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

class TestForward:
    def test_forward_calls_eval_fn_with_session(self):
        sim = _mock_sim()
        calls = []
        def eval_fn(session):
            calls.append(session)
            return 42.0
        adapter = OnnxQuantSimAdapter(sim, eval_fn=eval_fn)
        result = adapter.forward(batch=None)
        assert result == 42.0
        assert calls == [sim.session]


# ---------------------------------------------------------------------------
# state_bytes
# ---------------------------------------------------------------------------

class TestStateBytes:
    def test_state_bytes_returns_valid_tarball(self):
        """state_bytes() should produce a tar.gz with the three expected members."""
        sim = _mock_sim()

        def fake_export(path, filename_prefix, export_model):
            # Write stub files that sim.export() would produce
            with open(os.path.join(path, f"{filename_prefix}.onnx"), "wb") as f:
                f.write(b"onnx_data")
            with open(os.path.join(path, f"{filename_prefix}.encodings"), "wb") as f:
                f.write(b"enc_data")

        sim.export.side_effect = fake_export

        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0, qscheme_key="w4a8")
        data = adapter.state_bytes()

        assert isinstance(data, bytes)
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
        assert "worker_sim.onnx" in names
        assert "worker_sim.encodings" in names
        assert "metadata.json" in names

    def test_state_bytes_metadata_contains_qscheme(self):
        sim = _mock_sim()

        def fake_export(path, filename_prefix, export_model):
            with open(os.path.join(path, f"{filename_prefix}.onnx"), "wb") as f:
                f.write(b"x")
            with open(os.path.join(path, f"{filename_prefix}.encodings"), "wb") as f:
                f.write(b"x")

        sim.export.side_effect = fake_export

        adapter = OnnxQuantSimAdapter(
            sim, eval_fn=lambda s: 0.0, qscheme_key="fp16", config_file="default"
        )
        data = adapter.state_bytes()

        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            meta = json.loads(tar.extractfile("metadata.json").read())
        assert meta["qscheme_key"] == "fp16"
        assert meta["config_file"] == "default"

    def test_state_bytes_handles_encodings_json_extension(self):
        """AIMET may write .encodings.json instead of .encodings."""
        sim = _mock_sim()

        def fake_export(path, filename_prefix, export_model):
            with open(os.path.join(path, f"{filename_prefix}.onnx"), "wb") as f:
                f.write(b"x")
            # Write .encodings.json (not .encodings)
            with open(os.path.join(path, f"{filename_prefix}.encodings.json"), "wb") as f:
                f.write(b"enc_json")

        sim.export.side_effect = fake_export

        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        data = adapter.state_bytes()  # must not raise

        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            assert "worker_sim.encodings" in tar.getnames()

    def test_state_bytes_raises_if_no_onnx(self):
        sim = _mock_sim()
        sim.export.side_effect = lambda **kw: None  # produces nothing

        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        with pytest.raises(RuntimeError, match="worker_sim.onnx"):
            adapter.state_bytes()

    def test_state_bytes_raises_if_no_encodings(self):
        sim = _mock_sim()

        def fake_export(path, filename_prefix, export_model):
            with open(os.path.join(path, f"{filename_prefix}.onnx"), "wb") as f:
                f.write(b"x")
            # No encodings file

        sim.export.side_effect = fake_export

        adapter = OnnxQuantSimAdapter(sim, eval_fn=lambda s: 0.0)
        with pytest.raises(RuntimeError, match=".encodings"):
            adapter.state_bytes()


# ---------------------------------------------------------------------------
# from_state_bytes
# ---------------------------------------------------------------------------

class TestFromStateBytes:
    def _make_adapter_from_tarball(self, tarball_bytes, eval_fn=None):
        """Reconstruct adapter with all AIMET/onnx imports mocked."""
        if eval_fn is None:
            eval_fn = lambda s: 0.0

        mock_onnx_model = MagicMock()
        mock_sim = MagicMock()
        mock_sim.session = MagicMock()

        with patch("octopus.adapters.onnx_quantsim._QSCHEME_TYPES", None), \
             patch.dict("sys.modules", {
                 "aimet_common": MagicMock(),
                 "aimet_common.defs": MagicMock(),
                 "aimet_onnx": MagicMock(),
                 "aimet_onnx.quantsim": MagicMock(),
                 "aimet_onnx.utils": MagicMock(),
                 "onnx": MagicMock(),
             }):
            import sys
            # Set up return values
            sys.modules["onnx"].load.return_value = mock_onnx_model
            sys.modules["aimet_onnx.quantsim"].QuantizationSimModel.return_value = mock_sim
            sys.modules["aimet_common.defs"].QuantScheme.post_training_tf = "tf"
            sys.modules["aimet_common.defs"].QuantizationDataType.int = "int"
            sys.modules["aimet_common.defs"].QuantizationDataType.float = "float"

            adapter = OnnxQuantSimAdapter.from_state_bytes(tarball_bytes, eval_fn)

        return adapter, mock_sim

    def test_from_state_bytes_returns_adapter(self):
        tarball = _make_tarball()
        adapter, _ = self._make_adapter_from_tarball(tarball)
        assert isinstance(adapter, OnnxQuantSimAdapter)
        assert adapter.model_type_name == "onnx_quantsim"

    def test_from_state_bytes_preserves_qscheme_key(self):
        tarball = _make_tarball(meta={"qscheme_key": "fp16", "config_file": "htp_v81"})
        adapter, _ = self._make_adapter_from_tarball(tarball)
        assert adapter._qscheme_key == "fp16"

    def test_from_state_bytes_preserves_config_file(self):
        tarball = _make_tarball(meta={"qscheme_key": "w8a8", "config_file": "custom_cfg"})
        adapter, _ = self._make_adapter_from_tarball(tarball)
        assert adapter._config_file == "custom_cfg"

    def test_from_state_bytes_calls_load_encodings_strict_false(self):
        """strict=False is critical for mixed-precision encodings."""
        tarball = _make_tarball()
        eval_fn = lambda s: 0.0

        mock_sim = MagicMock()
        mock_sim.session = MagicMock()

        with patch.dict("sys.modules", {
            "aimet_common": MagicMock(),
            "aimet_common.defs": MagicMock(),
            "aimet_onnx": MagicMock(),
            "aimet_onnx.quantsim": MagicMock(),
            "aimet_onnx.utils": MagicMock(),
            "onnx": MagicMock(),
        }):
            import sys
            sys.modules["onnx"].load.return_value = MagicMock()
            sys.modules["aimet_onnx.quantsim"].QuantizationSimModel.return_value = mock_sim
            sys.modules["aimet_common.defs"].QuantScheme.post_training_tf = "tf"
            sys.modules["aimet_common.defs"].QuantizationDataType.int = "int"
            sys.modules["aimet_common.defs"].QuantizationDataType.float = "float"

            load_enc = sys.modules["aimet_onnx.utils"].load_encodings_to_sim
            OnnxQuantSimAdapter.from_state_bytes(tarball, eval_fn)

            # Verify strict=False was passed
            call_kwargs = load_enc.call_args
            assert call_kwargs is not None
            # strict=False can be positional or keyword
            args, kwargs = call_kwargs
            assert kwargs.get("strict") is False or (len(args) >= 3 and args[2] is False)

    def test_from_state_bytes_uses_cuda_device_0(self):
        """Workers always see device 0 via CUDA_VISIBLE_DEVICES."""
        tarball = _make_tarball()
        eval_fn = lambda s: 0.0

        mock_sim = MagicMock()
        mock_sim.session = MagicMock()

        with patch.dict("sys.modules", {
            "aimet_common": MagicMock(),
            "aimet_common.defs": MagicMock(),
            "aimet_onnx": MagicMock(),
            "aimet_onnx.quantsim": MagicMock(),
            "aimet_onnx.utils": MagicMock(),
            "onnx": MagicMock(),
        }):
            import sys
            sys.modules["onnx"].load.return_value = MagicMock()
            QuantSim = sys.modules["aimet_onnx.quantsim"].QuantizationSimModel
            QuantSim.return_value = mock_sim
            sys.modules["aimet_common.defs"].QuantScheme.post_training_tf = "tf"
            sys.modules["aimet_common.defs"].QuantizationDataType.int = "int"
            sys.modules["aimet_common.defs"].QuantizationDataType.float = "float"

            OnnxQuantSimAdapter.from_state_bytes(tarball, eval_fn)

            call_kwargs = QuantSim.call_args[1]
            providers = call_kwargs.get("providers", QuantSim.call_args[0][4] if len(QuantSim.call_args[0]) > 4 else None)
            # providers should include CUDAExecutionProvider with device_id=0
            assert providers is not None
            cuda_provider = next(
                (p for p in providers if isinstance(p, tuple) and "CUDA" in p[0]),
                None,
            )
            assert cuda_provider is not None
            assert cuda_provider[1].get("device_id") == 0
