"""Unit tests for merge_encoding_dicts — pure Python, no GPU, no AIMET."""
from __future__ import annotations

import pytest

from octopus_aimet._merge_encodings import merge_encoding_dicts


def _enc(act: dict | None = None, param: dict | None = None) -> dict:
    """Build a minimal AIMET encoding dict."""
    return {
        "activation_encodings": act or {},
        "param_encodings": param or {},
        "version": "1.0",
    }


class TestMergeEmpty:
    def test_empty_list_returns_empty(self):
        assert merge_encoding_dicts([]) == {}

    def test_single_dict_returns_copy(self):
        d = _enc(act={"t": {"min": -1.0, "max": 1.0, "bitwidth": 8}})
        result = merge_encoding_dicts([d])
        assert result["activation_encodings"]["t"]["min"] == -1.0
        assert result["activation_encodings"]["t"]["max"] == 1.0


class TestActivationEncodings:
    def test_min_is_global_min(self):
        """global_min = min(worker_mins)"""
        d1 = _enc(act={"relu": {"min": -0.5, "max": 2.0, "bitwidth": 8}})
        d2 = _enc(act={"relu": {"min": -1.5, "max": 1.8, "bitwidth": 8}})
        d3 = _enc(act={"relu": {"min": -0.2, "max": 2.5, "bitwidth": 8}})
        merged = merge_encoding_dicts([d1, d2, d3])
        assert merged["activation_encodings"]["relu"]["min"] == pytest.approx(-1.5)

    def test_max_is_global_max(self):
        """global_max = max(worker_maxes)"""
        d1 = _enc(act={"relu": {"min": -0.5, "max": 2.0, "bitwidth": 8}})
        d2 = _enc(act={"relu": {"min": -1.5, "max": 1.8, "bitwidth": 8}})
        d3 = _enc(act={"relu": {"min": -0.2, "max": 2.5, "bitwidth": 8}})
        merged = merge_encoding_dicts([d1, d2, d3])
        assert merged["activation_encodings"]["relu"]["max"] == pytest.approx(2.5)

    def test_non_minmax_fields_preserved_from_first(self):
        """bitwidth and other scalar fields carry over from first occurrence."""
        d1 = _enc(act={"x": {"min": -1.0, "max": 1.0, "bitwidth": 8, "offset": 0}})
        d2 = _enc(act={"x": {"min": -2.0, "max": 0.5, "bitwidth": 8, "offset": 0}})
        merged = merge_encoding_dicts([d1, d2])
        assert merged["activation_encodings"]["x"]["bitwidth"] == 8

    def test_tensor_present_in_only_some_workers(self):
        """Tensor from a partial-overlap worker still appears in merged result."""
        d1 = _enc(act={"a": {"min": -1.0, "max": 1.0}})
        d2 = _enc(act={"a": {"min": -2.0, "max": 0.5}, "b": {"min": -0.1, "max": 0.1}})
        merged = merge_encoding_dicts([d1, d2])
        assert "b" in merged["activation_encodings"]
        assert merged["activation_encodings"]["a"]["min"] == pytest.approx(-2.0)


class TestParamEncodings:
    def test_per_channel_min_max_merged(self):
        """Per-channel params: per-index min/max merge."""
        d1 = _enc(param={"conv.weight": [
            {"min": -0.5, "max": 0.5},
            {"min": -0.3, "max": 0.3},
        ]})
        d2 = _enc(param={"conv.weight": [
            {"min": -0.8, "max": 0.4},  # wider min
            {"min": -0.2, "max": 0.6},  # wider max
        ]})
        merged = merge_encoding_dicts([d1, d2])
        channels = merged["param_encodings"]["conv.weight"]
        assert channels[0]["min"] == pytest.approx(-0.8)
        assert channels[0]["max"] == pytest.approx(0.5)
        assert channels[1]["min"] == pytest.approx(-0.3)
        assert channels[1]["max"] == pytest.approx(0.6)

    def test_per_channel_first_dict_sets_baseline(self):
        """First dict's channels are not clobbered if second has fewer."""
        d1 = _enc(param={"w": [
            {"min": -1.0, "max": 1.0},
            {"min": -0.5, "max": 0.5},
            {"min": -0.2, "max": 0.2},
        ]})
        d2 = _enc(param={"w": [
            {"min": -1.5, "max": 0.8},
            # only 2 channels in worker 2 — index 2 should not change
        ]})
        merged = merge_encoding_dicts([d1, d2])
        channels = merged["param_encodings"]["w"]
        assert len(channels) == 3
        assert channels[2]["min"] == pytest.approx(-0.2)  # unchanged
        assert channels[2]["max"] == pytest.approx(0.2)

    def test_param_not_in_first_worker(self):
        """Param appearing first in worker 2 is still captured."""
        d1 = _enc(param={})
        d2 = _enc(param={"new_layer": [{"min": -1.0, "max": 1.0}]})
        merged = merge_encoding_dicts([d1, d2])
        assert "new_layer" in merged["param_encodings"]


class TestVersionField:
    def test_version_from_first_dict(self):
        d1 = _enc()
        d1["version"] = "2.0"
        d2 = _enc()
        d2["version"] = "1.5"
        merged = merge_encoding_dicts([d1, d2])
        assert merged["version"] == "2.0"

    def test_default_version_fallback(self):
        d = {"activation_encodings": {}, "param_encodings": {}}  # no version key
        merged = merge_encoding_dicts([d])
        assert merged["version"] == "1.0"


class TestMultipleWorkersMerge:
    def test_four_workers_correctness(self):
        """Simulate 4-worker calibration with distinct min/max per worker."""
        worker_mins = [-0.1, -0.5, -0.3, -0.9]
        worker_maxes = [1.0, 0.8, 1.2, 0.7]
        dicts = [
            _enc(act={"out": {"min": mn, "max": mx, "bitwidth": 8}})
            for mn, mx in zip(worker_mins, worker_maxes)
        ]
        merged = merge_encoding_dicts(dicts)
        assert merged["activation_encodings"]["out"]["min"] == pytest.approx(min(worker_mins))
        assert merged["activation_encodings"]["out"]["max"] == pytest.approx(max(worker_maxes))
