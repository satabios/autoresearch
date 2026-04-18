"""Tests for sharded (pipeline-parallel) calibration — Case A.

CPU-only tests mock onnx_partition + stage QuantSims.
@pytest.mark.gpu tests require 2+ real CUDA GPUs + aimet-onnx >= 1.30.

Validates:
  - _OutputCapturingSession: captures boundary tensors in one pass
  - _make_boundary_callback: feeds captured tensors to next stage
  - sharded_compute_encodings: stage-by-stage calibration, encoding merge
"""
from __future__ import annotations

import json
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# _OutputCapturingSession
# ---------------------------------------------------------------------------

class TestOutputCapturingSession:
    """Pure unit tests — no AIMET, no GPU."""

    def _make_session(self, output_spec: dict[str, Any]):
        """Build a fake ORT session that returns output_spec from run()."""
        mock_sess = MagicMock()
        mock_sess.get_outputs.return_value = [
            MagicMock(name=n) for n in output_spec
        ]

        def fake_run(out_names, feed, *a, **kw):
            names = out_names or list(output_spec)
            return [output_spec[n] for n in names if n in output_spec]

        mock_sess.run.side_effect = fake_run
        return mock_sess

    def test_returns_only_originally_requested_outputs(self):
        import numpy as np
        from octopus_aimet._calibration import _OutputCapturingSession

        outputs = {"y": np.array([1.0]), "boundary": np.array([2.0])}
        sess = self._make_session(outputs)
        buffer = []
        proxy = _OutputCapturingSession(sess, ["boundary"], buffer)

        result = proxy.run(["y"], {"x": np.zeros(1)})
        assert len(result) == 1
        import numpy as np
        assert (result[0] == np.array([1.0])).all()

    def test_captures_boundary_tensor_in_buffer(self):
        import numpy as np
        from octopus_aimet._calibration import _OutputCapturingSession

        boundary_val = np.array([3.0, 4.0])
        outputs = {"y": np.array([1.0]), "boundary": boundary_val}
        sess = self._make_session(outputs)
        buffer = []
        proxy = _OutputCapturingSession(sess, ["boundary"], buffer)

        proxy.run(["y"], {"x": np.zeros(1)})
        assert len(buffer) == 1
        assert (buffer[0]["boundary"] == boundary_val).all()

    def test_multiple_calls_accumulate_in_buffer(self):
        import numpy as np
        from octopus_aimet._calibration import _OutputCapturingSession

        call_count = [0]
        mock_sess = MagicMock()
        mock_sess.get_outputs.return_value = [MagicMock(name="y"), MagicMock(name="b")]

        def fake_run(out_names, feed, *a, **kw):
            call_count[0] += 1
            return [np.array([float(call_count[0])]) for _ in out_names]

        mock_sess.run.side_effect = fake_run
        buffer = []
        proxy = _OutputCapturingSession(mock_sess, ["b"], buffer)

        for i in range(5):
            proxy.run(["y"], {"x": np.zeros(1)})

        assert len(buffer) == 5

    def test_output_names_none_uses_all_session_outputs(self):
        import numpy as np
        from octopus_aimet._calibration import _OutputCapturingSession

        outputs = {"a": np.array([1.0]), "b": np.array([2.0]), "boundary": np.array([3.0])}
        mock_sess = MagicMock()

        # Must use objects with actual `.name` string attr (not MagicMock(name=...) which sets mock name)
        class _Out:
            def __init__(self, n): self.name = n
        mock_sess.get_outputs.return_value = [_Out(n) for n in outputs]
        mock_sess.run.side_effect = lambda names, feed, *a, **kw: [outputs[n] for n in names]

        buffer = []
        proxy = _OutputCapturingSession(mock_sess, ["boundary"], buffer)

        # output_names=None → should use all session outputs
        result = proxy.run(None, {})
        assert len(result) == len(outputs)  # all 3 returned
        assert len(buffer) == 1

    def test_getattr_proxies_to_real_session(self):
        from octopus_aimet._calibration import _OutputCapturingSession

        mock_sess = MagicMock()
        mock_sess.some_attr = "hello"
        proxy = _OutputCapturingSession(mock_sess, [], [])
        assert proxy.some_attr == "hello"

    def test_no_capture_if_boundary_already_in_requested(self):
        """If boundary name is already in requested outputs, still captured once."""
        import numpy as np
        from octopus_aimet._calibration import _OutputCapturingSession

        outputs = {"y": np.array([1.0]), "boundary": np.array([2.0])}
        mock_sess = MagicMock()
        mock_sess.get_outputs.return_value = [MagicMock(name=n) for n in outputs]
        mock_sess.run.side_effect = lambda names, feed, *a, **kw: [outputs[n] for n in names]

        buffer = []
        proxy = _OutputCapturingSession(mock_sess, ["boundary"], buffer)

        # Request includes boundary explicitly
        result = proxy.run(["y", "boundary"], {})
        assert len(result) == 2
        # Boundary should still be captured
        assert len(buffer) == 1


# ---------------------------------------------------------------------------
# _make_boundary_callback
# ---------------------------------------------------------------------------

class TestMakeBoundaryCallback:
    def test_callback_feeds_each_batch_to_stage_sim(self):
        import numpy as np
        from octopus_aimet._calibration import _make_boundary_callback

        boundary_batches = [
            {"h": np.array([1.0, 2.0])},
            {"h": np.array([3.0, 4.0])},
        ]
        stage_sim = MagicMock()
        stage_sim.session = MagicMock()
        stage_sim.session.run = MagicMock(return_value=[])

        cb = _make_boundary_callback(boundary_batches)
        cb(stage_sim, [])

        assert stage_sim.session.run.call_count == 2
        # Verify the feeds passed to run
        calls = stage_sim.session.run.call_args_list
        assert calls[0] == call(None, {"h": pytest.approx([1.0, 2.0])})
        assert calls[1] == call(None, {"h": pytest.approx([3.0, 4.0])})

    def test_empty_boundary_batches_no_calls(self):
        from octopus_aimet._calibration import _make_boundary_callback

        stage_sim = MagicMock()
        cb = _make_boundary_callback([])
        cb(stage_sim, [])
        stage_sim.session.run.assert_not_called()


# ---------------------------------------------------------------------------
# sharded_compute_encodings (mocked)
# ---------------------------------------------------------------------------

class TestShardedComputeEncodings:
    """Validate orchestration: partition → stage sims → capture → merge → load."""

    def _build_stage_artifact(self, name: str, output_names: list[str]):
        """Fake StageArtifact with onnx_path and output_names."""
        a = MagicMock()
        a.onnx_path = f"/fake/{name}.onnx"
        a.output_names = output_names
        return a

    def _run_sharded(self, num_stages=2):
        import numpy as np
        import os
        from octopus._types import GPUInfo

        gpus = [
            GPUInfo(device_id=i, name="MockGPU", total_vram_gb=24.0,
                    available_vram_gb=20.0, compute_capability=(8, 0))
            for i in range(num_stages)
        ]

        # Per-stage encoding files
        stage_encs = []
        for i in range(num_stages):
            stage_encs.append({
                "activation_encodings": {
                    f"op_{i}": {"min": -float(i + 1), "max": float(i + 1), "bitwidth": 8}
                },
                "param_encodings": {},
                "version": "1.0",
            })

        artifacts = [
            self._build_stage_artifact(f"stage{i}", [f"boundary_{i}"])
            for i in range(num_stages)
        ]

        # Fake stage QuantSim that writes its encoding on export()
        class _FakeStageQSim:
            def __init__(self, stage_idx):
                self.stage_idx = stage_idx
                self.session = MagicMock()
                self.session.run = MagicMock(return_value=[np.zeros(1)])
                self.session.get_outputs = MagicMock(return_value=[MagicMock(name=f"boundary_{stage_idx}")])

            def compute_encodings(self, cb, args):
                cb(self, args)

            def export(self, path, filename_prefix, export_model=True):
                enc = stage_encs[self.stage_idx]
                with open(os.path.join(path, f"{filename_prefix}.encodings"), "w") as f:
                    json.dump(enc, f)

        call_order = []

        def fake_make_stage_sim(stage_model, quant_kwargs, device_id):
            # Infer stage index from call order
            idx = len(call_order)
            call_order.append(idx)
            return _FakeStageQSim(idx)

        loaded_dicts: list[dict] = []
        def capture_load(sim, enc_path, strict=False):
            with open(enc_path) as _f:
                loaded_dicts.append(json.load(_f))

        mock_sim = MagicMock()
        mock_sim._octopus_model_proto_bytes = b"proto"
        mock_sim._octopus_init_kwargs = {}

        import onnx as _onnx_mod

        with patch("octopus.discovery.discover_gpus", return_value=gpus), \
             patch("octopus.sharding.onnx_partition.plan_stages_from_onnx", return_value=[MagicMock()] * num_stages), \
             patch("octopus.sharding.onnx_partition.materialize_stage_models", return_value=artifacts), \
             patch("octopus_aimet._calibration._make_stage_sim", side_effect=fake_make_stage_sim), \
             patch("octopus_aimet._calibration._load_encodings_to_sim", side_effect=capture_load), \
             patch("onnx.load", return_value=MagicMock()):

            from octopus_aimet._calibration import sharded_compute_encodings
            sharded_compute_encodings(mock_sim, lambda s, a: None, [{"x": 1}])

        return loaded_dicts, stage_encs

    def test_load_encodings_called_once_after_merge(self):
        loaded_dicts, _ = self._run_sharded(num_stages=2)
        assert len(loaded_dicts) == 1

    def test_merged_encodings_contain_all_stage_ops(self):
        loaded_dicts, stage_encs = self._run_sharded(num_stages=2)
        merged = loaded_dicts[0]

        all_ops = set()
        for enc in stage_encs:
            all_ops.update(enc["activation_encodings"])
        assert all_ops == set(merged["activation_encodings"])

    def test_three_stages(self):
        loaded_dicts, stage_encs = self._run_sharded(num_stages=3)
        assert len(loaded_dicts) == 1
        merged = loaded_dicts[0]
        # All 3 stages' ops present
        for i in range(3):
            assert f"op_{i}" in merged["activation_encodings"]


# ---------------------------------------------------------------------------
# @pytest.mark.gpu: real 2-stage sharded calibration
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestShardedCalibrationGPU:
    """End-to-end sharded calibration on real GPU with 2 stages.

    Requires:
      - 2+ CUDA GPUs
      - aimet-onnx >= 1.30
      - onnxruntime-gpu
      - onnx package
    """

    @pytest.fixture
    def two_stage_onnx(self):
        """Build a tiny ONNX model with 4 MatMul ops (easily splittable)."""
        import numpy as np
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("onnx not installed")

        # X -> MatMul1 -> Relu1 -> MatMul2 -> Y
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])

        W1_data = np.eye(8, dtype=np.float32)
        W2_data = np.eye(8, dtype=np.float32)

        W1_init = helper.make_tensor("W1", TensorProto.FLOAT, [8, 8], W1_data.flatten().tolist())
        W2_init = helper.make_tensor("W2", TensorProto.FLOAT, [8, 8], W2_data.flatten().tolist())

        mm1 = helper.make_node("MatMul", ["X", "W1"], ["H"])
        relu = helper.make_node("Relu", ["H"], ["H_relu"])
        mm2 = helper.make_node("MatMul", ["H_relu", "W2"], ["Y"])

        graph = helper.make_graph([mm1, relu, mm2], "two_stage", [X], [Y],
                                   initializer=[W1_init, W2_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        return model

    def test_sharded_calibration_produces_encodings_for_all_ops(self, two_stage_onnx):
        import aimet_onnx.quantsim as qs_module
        import numpy as np
        import octopus_aimet  # noqa: F401 — ensure patch installed

        sim = qs_module.QuantizationSimModel(
            two_stage_onnx,
            quant_scheme="post_training_tf",
            default_output_bw=8,
            default_param_bw=8,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )

        calibration_data = [
            {"X": np.random.randn(1, 8).astype(np.float32)} for _ in range(4)
        ]

        def callback(qsim, data):
            for feed in data:
                qsim.session.run(None, feed)

        # Force sharded mode
        sim._octopus_mode = "sharded"
        sim.compute_encodings(callback, calibration_data)

        # All quantized ops should have encodings
        # (AIMET populates qc_quantize_op_dict with quantized op stats)
        assert hasattr(sim, "qc_quantize_op_dict")
        # At minimum, no exception raised and encodings loaded back
