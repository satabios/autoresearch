"""Tests for SensitivityWorker enabling-loop helpers (no Ray/GPU required)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from octopus.worker import (
    _build_op_name_to_quantizers,
    _disable_all_quantizers,
    _restore_quantizers,
    _run_enabling_loop,
    _snapshot_quantizers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quantizer(enabled=True, data_type="int8", bitwidth=8, op_mode="qdq"):
    q = MagicMock()
    q.enabled = enabled
    q.data_type = data_type
    q.bitwidth = bitwidth
    q.op_mode = op_mode
    return q


# ---------------------------------------------------------------------------
# _build_op_name_to_quantizers
# ---------------------------------------------------------------------------

class TestBuildOpNameToQuantizers:
    def test_builds_mapping_from_qc_dict(self):
        q0, q1 = _make_quantizer(), _make_quantizer()
        sim = MagicMock()
        sim.qc_quantize_op_dict = {"MatMul_0": [q0], "Conv_1": [q1]}
        mapping = _build_op_name_to_quantizers(sim)
        assert mapping == {"MatMul_0": [q0], "Conv_1": [q1]}

    def test_wraps_single_quantizer_in_list(self):
        q = _make_quantizer()
        sim = MagicMock()
        sim.qc_quantize_op_dict = {"Op_0": q}  # not a list
        mapping = _build_op_name_to_quantizers(sim)
        assert mapping["Op_0"] == [q]

    def test_returns_empty_if_no_qc_dict(self):
        sim = MagicMock(spec=[])  # no qc_quantize_op_dict
        mapping = _build_op_name_to_quantizers(sim)
        assert mapping == {}


# ---------------------------------------------------------------------------
# _snapshot_quantizers
# ---------------------------------------------------------------------------

class TestSnapshotQuantizers:
    def test_snapshot_captures_state(self):
        q = _make_quantizer(enabled=True, data_type="int8", bitwidth=8, op_mode="qdq")
        mapping = {"Op_0": [q]}
        snap = _snapshot_quantizers(mapping)
        assert id(q) in snap
        enabled, dt, bw, om = snap[id(q)]
        assert enabled is True
        assert dt == "int8"
        assert bw == 8
        assert om == "qdq"

    def test_snapshot_multiple_quantizers(self):
        q0 = _make_quantizer(enabled=True)
        q1 = _make_quantizer(enabled=False)
        mapping = {"Op_0": [q0, q1]}
        snap = _snapshot_quantizers(mapping)
        assert len(snap) == 2
        assert snap[id(q0)][0] is True
        assert snap[id(q1)][0] is False


# ---------------------------------------------------------------------------
# _disable_all_quantizers
# ---------------------------------------------------------------------------

class TestDisableAllQuantizers:
    def test_disables_all(self):
        q0 = _make_quantizer(enabled=True)
        q1 = _make_quantizer(enabled=True)
        _disable_all_quantizers({"Op_0": [q0], "Op_1": [q1]})
        assert q0.enabled is False
        assert q1.enabled is False


# ---------------------------------------------------------------------------
# _restore_quantizers
# ---------------------------------------------------------------------------

class TestRestoreQuantizers:
    def test_restores_to_snapshot(self):
        q = _make_quantizer(enabled=True, data_type="int8", bitwidth=8, op_mode="qdq")
        mapping = {"Op_0": [q]}
        snap = _snapshot_quantizers(mapping)

        # Mutate
        q.enabled = False
        q.bitwidth = 4

        _restore_quantizers(mapping, snap)
        assert q.enabled is True
        assert q.bitwidth == 8

    def test_restore_skips_unknown_quantizer(self):
        """Quantizers not in snapshot should be left untouched."""
        q = _make_quantizer(enabled=False)
        mapping = {"Op_0": [q]}
        snap = {}  # empty snapshot
        _restore_quantizers(mapping, snap)
        assert q.enabled is False  # unchanged


# ---------------------------------------------------------------------------
# _run_enabling_loop
# ---------------------------------------------------------------------------

class TestRunEnablingLoop:
    def _make_sim_with_session(self):
        sim = MagicMock()
        sim.session = MagicMock()
        return sim

    def test_returns_score_per_layer(self):
        sim = self._make_sim_with_session()
        q = _make_quantizer()
        mapping = {"MatMul_0": [q], "Conv_1": [q]}

        scores = {"MatMul_0": 45.0, "Conv_1": 38.0}
        call_count = [0]

        def eval_fn(session):
            layer = ["MatMul_0", "Conv_1"][call_count[0]]
            call_count[0] += 1
            return scores[layer]

        with patch("octopus.worker.OpMode", create=True) as MockOpMode:
            MockOpMode.quantizeDequantize = "qdq"
            with patch.dict("sys.modules", {"aimet_common": MagicMock(), "aimet_common.defs": MagicMock()}):
                import sys
                sys.modules["aimet_common.defs"].OpMode.quantizeDequantize = "qdq"
                results = _run_enabling_loop(sim, ["MatMul_0", "Conv_1"], mapping, eval_fn)

        assert set(results.keys()) == {"MatMul_0", "Conv_1"}

    def test_eval_fn_exception_yields_nan(self):
        sim = self._make_sim_with_session()
        q = _make_quantizer()
        mapping = {"BadLayer": [q]}

        def eval_fn(session):
            raise RuntimeError("eval failed")

        with patch.dict("sys.modules", {"aimet_common": MagicMock(), "aimet_common.defs": MagicMock()}):
            import sys
            sys.modules["aimet_common.defs"].OpMode.quantizeDequantize = "qdq"
            results = _run_enabling_loop(sim, ["BadLayer"], mapping, eval_fn)

        import math
        assert math.isnan(results["BadLayer"])

    def test_quantizers_restored_after_loop(self):
        """All quantizers should be back to original state after loop completes."""
        sim = self._make_sim_with_session()
        q = _make_quantizer(enabled=True, bitwidth=8)
        mapping = {"Op_0": [q]}

        eval_fn = lambda session: 1.0

        with patch.dict("sys.modules", {"aimet_common": MagicMock(), "aimet_common.defs": MagicMock()}):
            import sys
            sys.modules["aimet_common.defs"].OpMode.quantizeDequantize = "qdq"
            _run_enabling_loop(sim, ["Op_0"], mapping, eval_fn)

        assert q.enabled is True
        assert q.bitwidth == 8

    def test_empty_layer_list_returns_empty(self):
        sim = self._make_sim_with_session()
        mapping = {}
        eval_fn = lambda session: 99.0

        with patch.dict("sys.modules", {"aimet_common": MagicMock(), "aimet_common.defs": MagicMock()}):
            import sys
            sys.modules["aimet_common.defs"].OpMode.quantizeDequantize = "qdq"
            results = _run_enabling_loop(sim, [], mapping, eval_fn)

        assert results == {}

    def test_layer_not_in_mapping_still_evaluated(self):
        """Layer with no quantizers should still call eval_fn (all-disabled baseline)."""
        sim = self._make_sim_with_session()
        mapping = {}  # no quantizers registered

        calls = []
        def eval_fn(session):
            calls.append(1)
            return 0.0

        with patch.dict("sys.modules", {"aimet_common": MagicMock(), "aimet_common.defs": MagicMock()}):
            import sys
            sys.modules["aimet_common.defs"].OpMode.quantizeDequantize = "qdq"
            results = _run_enabling_loop(sim, ["UnknownLayer"], mapping, eval_fn)

        assert "UnknownLayer" in results
        assert len(calls) == 1
