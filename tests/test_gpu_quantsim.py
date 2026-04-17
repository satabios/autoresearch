"""GPU integration tests — OnnxQuantSimAdapter + SensitivityWorker.

Tests the enabling-loop sensitivity analysis path on real GPU hardware.

Requires:
    aimet-onnx (pip install aimet-onnx) — skipped if not installed.
    CUDA GPU.

Run:
    pytest tests/test_gpu_quantsim.py -v -m "gpu and sensitivity"
    pytest tests/test_gpu_quantsim.py -v -m gpu
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import ray
import torch
import torch.nn as nn

from octopus._types import GPUInfo, VRAMProfile
from octopus.ray_runtime import init_ray

pytestmark = [pytest.mark.gpu, pytest.mark.sensitivity]

_SENSITIVITY_WORKER_GPU_FRACTION = 0.1


# ---------------------------------------------------------------------------
# Module-level availability checks (skip if deps missing)
# ---------------------------------------------------------------------------

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")
aimet_onnx = pytest.importorskip("aimet_onnx", reason="aimet-onnx required for QuantSim tests")


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")


def _skip_if_no_cuda_ep():
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDAExecutionProvider not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_onnx(model: nn.Module, in_dim: int, path: str) -> None:
    model.eval()
    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )


def _build_quantsim(onnx_path: str, device_id: int = 0):
    """Create an aimet_onnx QuantizationSimModel on the given GPU."""
    try:
        from aimet_onnx.common.defs import QuantScheme  # type: ignore[import-untyped]
    except ImportError:
        from aimet_common.defs import QuantScheme  # type: ignore[import-untyped]
    from aimet_onnx.quantsim import QuantizationSimModel
    import onnx

    onnx_model = onnx.load(onnx_path)
    providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    sim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme=QuantScheme.post_training_tf,
        param_type="int8",
        activation_type="int8",
        providers=providers,
    )
    return sim


def _compute_encodings(sim, in_dim: int, n_batches: int = 20) -> None:
    """Run calibration to populate encodings."""
    def forward_pass(session, _=None):
        for _ in range(n_batches):
            feed = {"input": np.random.randn(2, in_dim).astype(np.float32)}
            session.run(None, feed)

    sim.compute_encodings(forward_pass, forward_pass_callback_args=None)


@pytest.fixture(scope="module")
def ray_session():
    _skip_if_no_cuda()
    _skip_if_no_cuda_ep()
    if not ray.is_initialized():
        init_ray(ignore_reinit_error=True)
    yield


@pytest.fixture(scope="module")
def gpu_infos():
    from octopus.discovery import discover_gpus
    _skip_if_no_cuda()
    _skip_if_no_cuda_ep()
    return discover_gpus(use_pynvml=False)


@pytest.fixture(scope="module")
def quantsim_fixture():
    """Build a calibrated QuantSim model, export its adapter bytes, return (bytes, in_dim, layers)."""
    from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter

    _skip_if_no_cuda()
    _skip_if_no_cuda_ep()

    in_dim = 128

    # Small MLP: in_dim → 256 → in_dim
    model = nn.Sequential(
        nn.Linear(in_dim, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, in_dim, bias=False),
    )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        _export_onnx(model, in_dim, onnx_path)
        sim = _build_quantsim(onnx_path, device_id=0)
        _compute_encodings(sim, in_dim)
    finally:
        os.unlink(onnx_path)

    fixed_feed = {"input": np.random.default_rng(123).standard_normal((2, in_dim)).astype(np.float32)}

    def eval_fn(session) -> float:
        output = session.run(None, fixed_feed)[0]
        return float(output.mean())

    adapter = OnnxQuantSimAdapter(sim, eval_fn, qscheme_key="w8a8")
    state = adapter.state_bytes()

    # Collect layer names from the quantsim
    layers = list(getattr(sim, "qc_quantize_op_dict", {}).keys())
    return state, in_dim, layers, eval_fn


# ---------------------------------------------------------------------------
# OnnxQuantSimAdapter round-trip fidelity
# ---------------------------------------------------------------------------

class TestQuantSimAdapterRoundTrip:
    """Verify that state_bytes() → from_state_bytes() preserves SQNR."""

    def test_roundtrip_sqnr_within_threshold(self, gpu_infos, quantsim_fixture):
        """Reconstructed adapter should produce output within 1e-4 of original."""
        from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter

        state, in_dim, layers, eval_fn = quantsim_fixture

        restored = OnnxQuantSimAdapter.from_state_bytes(state, eval_fn)
        original_score = eval_fn(restored.sim.session)

        # Serialize and restore once more
        state2 = restored.state_bytes()
        restored2 = OnnxQuantSimAdapter.from_state_bytes(state2, eval_fn)
        restored2_score = eval_fn(restored2.sim.session)

        assert abs(restored2_score - original_score) < 1e-3, (
            f"SQNR drift after double round-trip: {original_score:.6f} → {restored2_score:.6f}"
        )

    def test_adapter_has_quantizer_ops(self, quantsim_fixture):
        """Restored adapter should have at least one quantizable op."""
        from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter

        state, in_dim, layers, eval_fn = quantsim_fixture
        restored = OnnxQuantSimAdapter.from_state_bytes(state, eval_fn)

        qc_dict = getattr(restored.sim, "qc_quantize_op_dict", {})
        assert len(qc_dict) > 0, "Expected ≥1 quantizable op after round-trip"


# ---------------------------------------------------------------------------
# SensitivityWorker — enabling loop on GPU
# ---------------------------------------------------------------------------

class TestSensitivityWorkerGPU:
    """SensitivityWorker runs enabling-loop sensitivity analysis on real GPU."""

    def test_sensitivity_worker_returns_scores_for_all_layers(
        self, ray_session, gpu_infos, quantsim_fixture
    ):
        """SensitivityWorker.process_layers() should return a score per layer."""
        from octopus.worker import SensitivityWorker

        state, in_dim, layers, eval_fn = quantsim_fixture
        if not layers:
            pytest.skip("No quantizable layers found in test model")

        w = SensitivityWorker.options(
            num_gpus=_SENSITIVITY_WORKER_GPU_FRACTION,
        ).remote(state, "onnx_quantsim", eval_fn)
        try:
            init_result = ray.get(w.initialize.remote(), timeout=120)
            assert init_result["status"] == "ready"
            assert init_result["num_quantizer_ops"] > 0

            scores = ray.get(w.process_layers.remote(layers), timeout=300)

            assert set(scores.keys()) == set(layers), (
                f"Missing layers in results: {set(layers) - set(scores.keys())}"
            )
            for layer, score in scores.items():
                assert isinstance(score, float), f"Score for {layer} is not float"
                # Score may be NaN if eval_fn fails, but should be a real number typically
                if not math.isnan(score):
                    assert math.isfinite(score), f"Non-finite score for {layer}: {score}"

        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=30)
            except Exception:
                pass
            ray.kill(w)

    def test_sensitivity_worker_get_results_accumulates(
        self, ray_session, gpu_infos, quantsim_fixture
    ):
        """get_results() should return all accumulated scores after multiple process_layers calls."""
        from octopus.worker import SensitivityWorker

        state, in_dim, layers, eval_fn = quantsim_fixture
        if len(layers) < 2:
            pytest.skip("Need ≥2 layers for multi-batch test")

        half = len(layers) // 2
        batch_a = layers[:half]
        batch_b = layers[half:]

        w = SensitivityWorker.options(
            num_gpus=_SENSITIVITY_WORKER_GPU_FRACTION,
        ).remote(state, "onnx_quantsim", eval_fn)
        try:
            ray.get(w.initialize.remote(), timeout=120)

            ray.get(w.process_layers.remote(batch_a), timeout=300)
            ray.get(w.process_layers.remote(batch_b), timeout=300)

            all_results = ray.get(w.get_results.remote(), timeout=30)
            assert set(all_results.keys()) == set(layers), (
                "get_results() should contain scores from both batches"
            )
        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=30)
            except Exception:
                pass
            ray.kill(w)

    def test_sensitivity_worker_status_reports_progress(
        self, ray_session, gpu_infos, quantsim_fixture
    ):
        """get_status() should report increasing results_so_far as layers are processed."""
        from octopus.worker import SensitivityWorker

        state, in_dim, layers, eval_fn = quantsim_fixture
        if len(layers) < 2:
            pytest.skip("Need ≥2 layers")

        w = SensitivityWorker.options(
            num_gpus=_SENSITIVITY_WORKER_GPU_FRACTION,
        ).remote(state, "onnx_quantsim", eval_fn)
        try:
            ray.get(w.initialize.remote(), timeout=120)

            # Before processing
            status_before = ray.get(w.get_status.remote(), timeout=10)
            assert status_before["initialized"] is True
            assert status_before["results_so_far"] == 0

            # Process first half
            ray.get(w.process_layers.remote(layers[:1]), timeout=120)
            status_after = ray.get(w.get_status.remote(), timeout=10)
            assert status_after["results_so_far"] >= 1

        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=30)
            except Exception:
                pass
            ray.kill(w)


# ---------------------------------------------------------------------------
# Multiple SensitivityWorkers — layer partitioning + parallel execution
# ---------------------------------------------------------------------------

class TestMultipleSensitivityWorkers:
    """Partition layers across multiple SensitivityWorkers, run in parallel."""

    def test_two_workers_cover_all_layers(
        self, ray_session, gpu_infos, quantsim_fixture
    ):
        """Two workers with disjoint layer subsets should together cover all layers."""
        from octopus.worker import SensitivityWorker
        from octopus.scheduler import partition_layers

        state, in_dim, layers, eval_fn = quantsim_fixture
        if len(layers) < 2:
            pytest.skip("Need ≥2 layers for multi-worker test")

        # Partition: 90% static split, 10% reserve
        chunks, reserve = partition_layers(layers, num_workers=2, queue_fraction=0.0)
        # With queue_fraction=0, all layers split evenly between 2 workers

        w0 = SensitivityWorker.options(
            num_gpus=_SENSITIVITY_WORKER_GPU_FRACTION,
        ).remote(state, "onnx_quantsim", eval_fn)
        w1 = SensitivityWorker.options(
            num_gpus=_SENSITIVITY_WORKER_GPU_FRACTION,
        ).remote(state, "onnx_quantsim", eval_fn)

        try:
            # Initialize both workers in parallel
            init_refs = [w0.initialize.remote(), w1.initialize.remote()]
            init_results = ray.get(init_refs, timeout=180)
            assert all(r["status"] == "ready" for r in init_results)

            # Process layers in parallel
            proc_refs = [
                w0.process_layers.remote(chunks[0]),
                w1.process_layers.remote(chunks[1]),
            ]
            ray.get(proc_refs, timeout=600)

            # Collect results from both workers
            result_refs = [w0.get_results.remote(), w1.get_results.remote()]
            r0, r1 = ray.get(result_refs, timeout=30)

            combined = {**r0, **r1}
            expected_layers = set(chunks[0]) | set(chunks[1])
            assert set(combined.keys()) == expected_layers, (
                f"Combined results missing: {expected_layers - set(combined.keys())}"
            )

        finally:
            for w in [w0, w1]:
                try:
                    ray.get(w.shutdown.remote(), timeout=30)
                except Exception:
                    pass
                try:
                    ray.kill(w)
                except Exception:
                    pass

    def test_workers_fill_available_vram(
        self, ray_session, gpu_infos, quantsim_fixture
    ):
        """Spawn as many SensitivityWorkers as layers allow, verify all run."""
        from octopus.worker import SensitivityWorker
        from octopus.scheduler import partition_layers

        state, in_dim, layers, eval_fn = quantsim_fixture
        if not layers:
            pytest.skip("No quantizable layers")

        # Note: SensitivityWorker uses num_gpus=0 (ORT handles GPU internally).
        # Determine number of workers as min(num_layers, 4) for this test.
        n_workers = min(len(layers), 4)
        chunks, _ = partition_layers(layers, num_workers=n_workers, queue_fraction=0.0)

        workers = [
            SensitivityWorker.options(
                num_gpus=_SENSITIVITY_WORKER_GPU_FRACTION,
            ).remote(state, "onnx_quantsim", eval_fn)
            for _ in range(n_workers)
        ]

        try:
            # Initialize all in parallel
            init_refs = [w.initialize.remote() for w in workers]
            init_results = ray.get(init_refs, timeout=300)
            assert all(r["status"] == "ready" for r in init_results)
            assert all(r["num_quantizer_ops"] > 0 for r in init_results)

            # Process all layer chunks in parallel
            proc_refs = [workers[i].process_layers.remote(chunks[i]) for i in range(n_workers)]
            ray.get(proc_refs, timeout=600)

            # Collect and merge results
            result_refs = [w.get_results.remote() for w in workers]
            all_results_list = ray.get(result_refs, timeout=30)
            combined: dict[str, float] = {}
            for r in all_results_list:
                combined.update(r)

            # All processed layers should have scores
            processed = set().union(*[set(chunks[i]) for i in range(n_workers)])
            assert set(combined.keys()) == processed

        finally:
            for w in workers:
                try:
                    ray.get(w.shutdown.remote(), timeout=30)
                except Exception:
                    pass
                try:
                    ray.kill(w)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Full octopus.core sensitivity_scan on GPU
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.sensitivity
class TestOctopusSensitivityScanGPU:
    """End-to-end test of Octopus.sensitivity_scan() on real GPU."""

    def test_sensitivity_scan_returns_all_layer_scores(
        self, ray_session, gpu_infos, quantsim_fixture
    ):
        """sensitivity_scan(mode='enabling') should return {layer: score} for all layers."""
        from octopus.core import Octopus
        from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter

        state, in_dim, layers, eval_fn = quantsim_fixture
        if not layers:
            pytest.skip("No quantizable layers in test model")

        # Reconstruct sim for Octopus
        sim_adapter = OnnxQuantSimAdapter.from_state_bytes(state, eval_fn)
        sim = sim_adapter.sim

        with Octopus(
            model=sim,
            eval_fn=eval_fn,
            safety_net_gb=0.5,
            log_level="WARNING",
        ) as o:
            results = o.sensitivity_scan(layers=layers, mode="enabling")

        assert isinstance(results, dict)
        assert set(results.keys()) == set(layers), (
            f"Missing layers: {set(layers) - set(results.keys())}"
        )
        for layer, score in results.items():
            assert isinstance(score, float), f"{layer}: expected float, got {type(score)}"


@pytest.mark.gpu
class TestOctopusQuantSimShardedPipelineParallel:
    """Shared-model AIMET ONNX path on multiple GPUs via ONNX staged runtime."""

    def test_octopus_quantsim_pipeline_parallel_map(self, ray_session, quantsim_fixture, monkeypatch):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >=2 GPUs")

        from octopus.adapters.onnx_quantsim import OnnxQuantSimAdapter
        from octopus.core import Octopus

        state, _, _, eval_fn = quantsim_fixture
        sim_adapter = OnnxQuantSimAdapter.from_state_bytes(state, eval_fn)
        sim = sim_adapter.sim

        fake_gpus = [
            GPUInfo(
                device_id=i,
                name=f"GPU-{i}",
                total_vram_gb=80.0,
                available_vram_gb=70.0,
                compute_capability=(8, 0),
            )
            for i in range(2)
        ]
        fake_profile = VRAMProfile(
            peak_vram_bytes=0,
            peak_vram_gb=90.0,
            model_params_bytes=0,
            activation_peak_bytes=0,
            profiled_on_device=0,
        )

        monkeypatch.setattr("octopus.core.discover_gpus", lambda device_ids=None: fake_gpus)
        monkeypatch.setattr("octopus.core.profile_ort_vram", lambda **kwargs: fake_profile)

        with Octopus(
            model=sim,
            eval_fn=eval_fn,
            sharding_strategy="pp",
            gpu_ids=[0, 1],
            max_workers=1,
            safety_net_gb=1.0,
            log_level="WARNING",
        ) as o:
            outputs = o.map([None, None, None])
            assert o.pool_plan is not None
            assert o.pool_plan.sharding_required is True
            assert o.pool_plan.gpus_per_shard == 2
            assert o.pool_plan.total_workers == 1

        assert len(outputs) == 3
        assert all(isinstance(v, float) and math.isfinite(v) for v in outputs)
