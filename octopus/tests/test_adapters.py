import io

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
