from __future__ import annotations

import tempfile

import numpy as np
import torch

from octopus.worker import _ORTPipelineSessionProxy, _normalize_ort_input_feed


class _FakeSession:
    def __init__(self, fn):
        self._fn = fn

    def run(self, output_names, input_feed):
        outputs = self._fn(input_feed)
        return [outputs[name] for name in output_names]


def test_normalize_ort_input_feed_from_tensor():
    x = torch.randn(2, 3)
    feed = _normalize_ort_input_feed(x, ("input",))

    assert set(feed.keys()) == {"input"}
    assert isinstance(feed["input"], np.ndarray)
    assert feed["input"].shape == (2, 3)


def test_onnx_pipeline_proxy_two_stage_run():
    def stage0(feed):
        return {"z": feed["x"] + 1.0}

    def stage1(feed):
        return {"y": feed["z"] * 2.0}

    proxy = _ORTPipelineSessionProxy(
        stage_sessions=[_FakeSession(stage0), _FakeSession(stage1)],
        stage_input_names=[("x",), ("z",)],
        stage_output_names=[("z",), ("y",)],
        temp_dirs=[tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()],
    )
    x = np.array([[1.0, 2.0]], dtype=np.float32)

    out = proxy.run(None, {"x": x})[0]

    np.testing.assert_allclose(out, (x + 1.0) * 2.0)
    proxy.close()


def test_onnx_pipeline_proxy_respects_requested_output_names():
    def stage0(feed):
        return {"z": feed["x"] + 3.0}

    def stage1(feed):
        return {"y": feed["z"] - 1.0, "y_alt": feed["z"] + 1.0}

    proxy = _ORTPipelineSessionProxy(
        stage_sessions=[_FakeSession(stage0), _FakeSession(stage1)],
        stage_input_names=[("x",), ("z",)],
        stage_output_names=[("z",), ("y", "y_alt")],
        temp_dirs=[],
    )
    x = np.array([[4.0]], dtype=np.float32)

    out = proxy.run(["y_alt"], {"x": x})[0]

    np.testing.assert_allclose(out, x + 4.0)
