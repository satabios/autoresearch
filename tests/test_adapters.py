import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from octopus.adapters import detect_and_wrap
from octopus.adapters.pytorch import PyTorchAdapter, _to_device


class TestPyTorchAdapter:
    def test_model_type_name(self, toy_model, toy_eval_fn):
        adapter = PyTorchAdapter(toy_model, toy_eval_fn)
        assert adapter.model_type_name == "pytorch"

    def test_state_bytes_roundtrip(self, toy_model, toy_eval_fn):
        """Serialize and deserialize should produce equivalent model."""
        adapter = PyTorchAdapter(toy_model, toy_eval_fn)
        data = adapter.state_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = PyTorchAdapter.from_state_bytes(data, toy_eval_fn)
        assert restored.model_type_name == "pytorch"

        # Compare weights
        original_sd = toy_model.state_dict()
        restored_sd = restored._original_model.state_dict()
        for key in original_sd:
            assert torch.allclose(original_sd[key], restored_sd[key])

    def test_eval_fn_property(self, toy_model, toy_eval_fn):
        adapter = PyTorchAdapter(toy_model, toy_eval_fn)
        assert adapter.eval_fn is toy_eval_fn


class TestToDevice:
    def test_tensor(self):
        t = torch.randn(3, 4)
        result = _to_device(t, torch.device("cpu"))
        assert isinstance(result, torch.Tensor)

    def test_dict_of_tensors(self):
        d = {"a": torch.randn(2, 3), "b": torch.randn(4)}
        result = _to_device(d, torch.device("cpu"))
        assert isinstance(result, dict)
        assert "a" in result and "b" in result

    def test_list_of_tensors(self):
        lst = [torch.randn(2), torch.randn(3)]
        result = _to_device(lst, torch.device("cpu"))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_non_tensor_passthrough(self):
        result = _to_device("hello", torch.device("cpu"))
        assert result == "hello"

    def test_none_device_passthrough(self):
        t = torch.randn(3)
        result = _to_device(t, None)
        assert torch.equal(result, t)


class TestDetectAndWrap:
    def test_detects_pytorch_module(self, toy_model, toy_eval_fn):
        adapter = detect_and_wrap(toy_model, toy_eval_fn)
        assert isinstance(adapter, PyTorchAdapter)
        assert adapter.model_type_name == "pytorch"

    def test_rejects_unknown_type(self, toy_eval_fn):
        with pytest.raises(TypeError, match="Unsupported model type"):
            detect_and_wrap("not a model", toy_eval_fn)

    def test_rejects_int(self, toy_eval_fn):
        with pytest.raises(TypeError):
            detect_and_wrap(42, toy_eval_fn)


class TestONNXRuntimeAdapter:
    def test_state_bytes_include_external_data(self, tmp_path):
        from octopus.adapters.onnx_rt import ONNXRuntimeAdapter

        model_path = tmp_path / "test.onnx"
        sidecar_path = tmp_path / "test.onnx.data"
        model_path.write_bytes(b"onnx-model")
        sidecar_path.write_bytes(b"external-data")

        fake_ort = MagicMock()
        fake_ort.InferenceSession = type("FakeSession", (), {})

        with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
            adapter = ONNXRuntimeAdapter(str(model_path), lambda s, b: b)
            state = adapter.state_bytes()

        with tarfile.open(fileobj=io.BytesIO(state), mode="r:gz") as tar:
            names = sorted(tar.getnames())

        assert names == ["test.onnx", "test.onnx.data"]

    def test_load_to_device_restores_external_data_sidecars(self, tmp_path):
        from octopus.adapters.onnx_rt import ONNXRuntimeAdapter

        model_path = tmp_path / "test.onnx"
        sidecar_path = tmp_path / "test.onnx.data"
        model_path.write_bytes(b"onnx-model")
        sidecar_path.write_bytes(b"external-data")

        class FakeSession:
            def __init__(self, path, providers):
                restored_model = Path(path)
                restored_sidecar = restored_model.with_name(restored_model.name + ".data")
                assert restored_model.exists()
                assert restored_sidecar.exists()
                assert restored_sidecar.read_bytes() == b"external-data"

        fake_ort = MagicMock()
        fake_ort.InferenceSession = FakeSession

        with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
            adapter = ONNXRuntimeAdapter(str(model_path), lambda s, b: b)
            restored = ONNXRuntimeAdapter.from_state_bytes(adapter.state_bytes(), lambda s, b: b)
            restored.load_to_device(torch.device("cuda:0"))
            restored.unload()
