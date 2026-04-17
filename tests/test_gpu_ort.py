"""GPU integration tests — ONNXRuntimeAdapter + InferenceWorker.

Exports small ONNX models from PyTorch, loads them onto GPU via ORT's
CUDAExecutionProvider, packs GPUs with K workers, runs all in parallel.

Run:
    pytest tests/test_gpu_ort.py -v -m gpu
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
import torch.nn as nn

from octopus._types import GPUInfo, VRAMProfile
from octopus.adapters.onnx_rt import ONNXRuntimeAdapter
from octopus.discovery import compute_worker_allocation, discover_gpus
from octopus.ray_runtime import init_ray
from octopus.worker import InferenceWorker, ShardedInferenceWorkerGroup

pytestmark = pytest.mark.gpu

# Skip entire module if onnxruntime not installed
ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT GPU tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")


def _skip_if_no_cuda_ep():
    """Skip if ORT CUDA execution provider is not available."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDAExecutionProvider not available (onnxruntime-gpu required)")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=False),
    )


def _export_onnx(model: nn.Module, in_dim: int, path: str) -> None:
    """Export model to ONNX with dynamic batch axis."""
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
def _cleanup_onnx_export(path: str) -> None:
    Path(path).unlink(missing_ok=True)
    Path(f"{path}.data").unlink(missing_ok=True)


def _export_adapter_state(model: nn.Module, in_dim: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        _export_onnx(model, in_dim, onnx_path)
        adapter = ONNXRuntimeAdapter(onnx_path, eval_fn=_ort_eval_fn)
        return adapter.state_bytes()
    finally:
        _cleanup_onnx_export(onnx_path)


def _model_vram_gb(model: nn.Module) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1 << 30)


def _ort_eval_fn(session, feed: dict) -> np.ndarray:
    """Eval fn for ONNXRuntimeAdapter: run session, return first output."""
    return session.run(None, feed)[0]


@pytest.fixture(scope="module")
def ray_session():
    _skip_if_no_cuda()
    _skip_if_no_cuda_ep()
    if not ray.is_initialized():
        init_ray(ignore_reinit_error=True)
    yield


@pytest.fixture(scope="module")
def gpu_infos():
    _skip_if_no_cuda()
    _skip_if_no_cuda_ep()
    return discover_gpus(use_pynvml=False)


# ---------------------------------------------------------------------------
# ONNXRuntimeAdapter CUDA execution provider sanity
# ---------------------------------------------------------------------------

class TestORTAdapterCUDA:
    """Verify ORT sessions use the CUDA execution provider."""

    def test_ort_cuda_ep_active_after_load(self, gpu_infos):
        """load_to_device() should create a session with CUDAExecutionProvider."""
        device_id = gpu_infos[0].device_id
        device = torch.device(f"cuda:{device_id}")

        model = nn.Linear(512, 512, bias=False)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _export_onnx(model, 512, onnx_path)
            adapter = ONNXRuntimeAdapter(onnx_path, eval_fn=_ort_eval_fn)
            adapter.load_to_device(device)

            assert adapter._session is not None
            active_providers = adapter._session.get_providers()
            assert any("CUDA" in p for p in active_providers), (
                f"Expected CUDAExecutionProvider in {active_providers}"
            )
            adapter.unload()
        finally:
            _cleanup_onnx_export(onnx_path)

    def test_ort_adapter_roundtrip_state_bytes(self, gpu_infos):
        """state_bytes() → from_state_bytes() should reconstruct without error."""
        model = nn.Linear(256, 128, bias=False)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _export_onnx(model, 256, onnx_path)
            adapter = ONNXRuntimeAdapter(onnx_path, eval_fn=_ort_eval_fn)
            data = adapter.state_bytes()

            restored = ONNXRuntimeAdapter.from_state_bytes(data, _ort_eval_fn)
            assert restored.model_type_name == "onnx"

            device = torch.device(f"cuda:{gpu_infos[0].device_id}")
            restored.load_to_device(device)
            active = restored._session.get_providers()
            assert any("CUDA" in p for p in active)
            restored.unload()
        finally:
            _cleanup_onnx_export(onnx_path)

    def test_ort_adapter_forward_output_correct(self, gpu_infos):
        """Forward pass on GPU should match CPU numpy output."""
        device_id = gpu_infos[0].device_id
        device = torch.device(f"cuda:{device_id}")

        H = 128
        model = nn.Linear(H, H, bias=False)
        torch.manual_seed(0)
        nn.init.xavier_uniform_(model.weight)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _export_onnx(model, H, onnx_path)
            adapter = ONNXRuntimeAdapter(onnx_path, eval_fn=_ort_eval_fn)
            adapter.load_to_device(device)

            # Build feed dict (ORT runs on CUDA internally but accepts CPU numpy)
            batch_np = np.random.randn(2, H).astype(np.float32)
            feed = {"input": batch_np}

            output = adapter.forward(feed)
            assert isinstance(output, np.ndarray)
            assert output.shape == (2, H)

            # Verify against torch CPU
            with torch.no_grad():
                expected = model(torch.from_numpy(batch_np)).numpy()
            np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)

            adapter.unload()
        finally:
            _cleanup_onnx_export(onnx_path)


# ---------------------------------------------------------------------------
# InferenceWorker with ORT — max K workers
# ---------------------------------------------------------------------------

class TestORTMaxKWorkers:
    """Pack GPUs with K ORT workers, run all in parallel, verify results."""

    def test_max_k_ort_workers_parallel_run(self, ray_session, gpu_infos):
        """Spawn K ORT workers (fill GPU VRAM), submit K*3 batches in parallel."""
        safety_net_gb = 0.5
        H = 2048
        model = _mlp(H, H * 2, H)
        # ORT doesn't register VRAM via torch.cuda; use model param size as estimate.
        # ORT sessions also allocate ~200-400 MB for CUDA context — apply 1.5× overhead.
        model_vram_gb = _model_vram_gb(model) * 1.5

        plan = compute_worker_allocation(
            gpu_infos,
            model_vram_gb=model_vram_gb,
            safety_net_gb=safety_net_gb,
            max_workers=min(max(2, len(gpu_infos) * 2), 8),
        )
        K = plan.total_workers
        assert K >= 1, "Should fit at least 1 ORT worker"

        model_bytes = _export_adapter_state(model, H)

        workers: list = []
        for alloc in plan.allocations:
            frac = 1.0 / alloc.num_workers
            for _ in range(alloc.num_workers):
                w = InferenceWorker.options(num_gpus=frac).remote(
                    model_bytes, "onnx", _ort_eval_fn
                )
                workers.append(w)

        try:
            init_refs = [w.initialize.remote() for w in workers]
            init_results = ray.get(init_refs, timeout=180)

            assert len(init_results) == K
            assert all(r["status"] == "ready" for r in init_results)
            # Note: memory_allocated_mb may be 0 for ORT (uses separate CUDA context from torch)

            # Submit K*3 batches: each batch is a numpy feed dict
            batches = [{"input": np.random.randn(2, H).astype(np.float32)} for _ in range(K * 3)]
            run_refs = [
                workers[i % K].run.remote(batches[i])
                for i in range(len(batches))
            ]
            results = ray.get(run_refs, timeout=120)

            assert len(results) == K * 3
            for r in results:
                assert isinstance(r, np.ndarray), f"Expected ndarray, got {type(r)}"
                assert r.shape == (2, H), f"Unexpected output shape {r.shape}"
                assert np.isfinite(r).all(), "Output contains non-finite values"

        finally:
            try:
                ray.get([w.shutdown.remote() for w in workers], timeout=30)
            except Exception:
                pass
            for w in workers:
                try:
                    ray.kill(w)
                except Exception:
                    pass

    def test_ort_worker_init_and_health_check(self, ray_session, gpu_infos):
        """InferenceWorker with ORT should initialize and pass health_check."""
        H = 256
        model = nn.Linear(H, H, bias=False)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _export_onnx(model, H, onnx_path)
            adapter = ONNXRuntimeAdapter(onnx_path, eval_fn=_ort_eval_fn)
            model_bytes = adapter.state_bytes()
        finally:
            _cleanup_onnx_export(onnx_path)

        w = InferenceWorker.options(num_gpus=0.1).remote(model_bytes, "onnx", _ort_eval_fn)

        try:
            result = ray.get(w.initialize.remote(), timeout=60)
            assert result["status"] == "ready"

            healthy = ray.get(w.health_check.remote(), timeout=15)
            assert healthy is True
        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=15)
            except Exception:
                pass
            ray.kill(w)

    def test_ort_worker_output_matches_cpu(self, ray_session, gpu_infos):
        """ORT CUDA output should match ORT CPU output to float32 tolerance."""
        H = 128
        model = nn.Linear(H, H, bias=False)
        torch.manual_seed(1)
        nn.init.xavier_uniform_(model.weight)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _export_onnx(model, H, onnx_path)
            cpu_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            batch_np = np.random.default_rng(42).standard_normal((2, H)).astype(np.float32)
            expected = cpu_sess.run(None, {"input": batch_np})[0]
            adapter = ONNXRuntimeAdapter(onnx_path, eval_fn=_ort_eval_fn)
            model_bytes = adapter.state_bytes()
        finally:
            _cleanup_onnx_export(onnx_path)
        w = InferenceWorker.options(num_gpus=0.1).remote(model_bytes, "onnx", _ort_eval_fn)

        try:
            ray.get(w.initialize.remote(), timeout=60)
            gpu_output = ray.get(w.run.remote({"input": batch_np}), timeout=30)

            np.testing.assert_allclose(gpu_output, expected, rtol=1e-3, atol=1e-4)
        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=15)
            except Exception:
                pass
            ray.kill(w)


# ---------------------------------------------------------------------------
# Multi-GPU ORT
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestORTMultiGPU:
    """ORT workers distributed across multiple GPUs."""

    def test_ort_workers_span_all_gpus(self, ray_session, gpu_infos):
        if len(gpu_infos) < 2:
            pytest.skip("Requires ≥2 GPUs")

        H = 1024
        model = nn.Linear(H, H, bias=False)
        model_vram_gb = _model_vram_gb(model) * 1.5  # ORT context overhead

        model_bytes = _export_adapter_state(model, H)

        plan = compute_worker_allocation(
            gpu_infos,
            model_vram_gb,
            safety_net_gb=0.5,
            workers_per_gpu=1,
        )
        assert len(plan.allocations) > 1, "Expected workers on >1 GPU"

        workers: list = []
        for alloc in plan.allocations:
            frac = 1.0 / alloc.num_workers
            for _ in range(alloc.num_workers):
                w = InferenceWorker.options(num_gpus=frac).remote(
                    model_bytes, "onnx", _ort_eval_fn
                )
                workers.append(w)

        try:
            init_refs = [w.initialize.remote() for w in workers]
            results = ray.get(init_refs, timeout=180)
            assert all(r["status"] == "ready" for r in results)

            K = len(workers)
            batches = [{"input": np.random.randn(1, H).astype(np.float32)} for _ in range(K)]
            run_refs = [workers[i].run.remote(batches[i]) for i in range(K)]
            run_results = ray.get(run_refs, timeout=60)
            assert len(run_results) == K
        finally:
            try:
                ray.get([w.shutdown.remote() for w in workers], timeout=30)
            except Exception:
                pass
            for w in workers:
                try:
                    ray.kill(w)
                except Exception:
                    pass


@pytest.mark.gpu
class TestORTShardedPipelineParallel:
    """Shared-model ONNX Runtime path on multiple GPUs via pipeline parallel."""

    def test_sharded_worker_group_pipeline_parallel_run(self, ray_session):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >=2 GPUs")

        H = 128
        model = _mlp(H, 2 * H, H)
        model_bytes = _export_adapter_state(model, H)
        feed = {"input": np.random.randn(2, H).astype(np.float32)}

        worker = ShardedInferenceWorkerGroup.options(num_gpus=2).remote(
            model_bytes,
            "onnx",
            _ort_eval_fn,
            [0, 1],
            "pp",
            0,
        )

        try:
            init_result = ray.get(worker.initialize.remote(), timeout=180)
            assert init_result["status"] == "ready"
            assert init_result["strategy"] == "pp"
            assert init_result["adapter"] == "onnx"
            assert init_result["visible_gpu_count"] == 2
            assert len(init_result["assigned_gpu_ids"]) == 2

            output = ray.get(worker.run.remote(feed), timeout=120)
            assert isinstance(output, np.ndarray)
            assert output.shape == (2, H)
            assert np.isfinite(output).all()
        finally:
            try:
                ray.get(worker.shutdown.remote(), timeout=30)
            except Exception:
                pass
            ray.kill(worker)

    def test_octopus_ort_pipeline_parallel_map(self, ray_session, monkeypatch):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >=2 GPUs")

        from octopus.core import Octopus

        H = 128
        model = _mlp(H, 2 * H, H)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        _export_onnx(model, H, onnx_path)

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

        inputs = [{"input": np.random.randn(2, H).astype(np.float32)} for _ in range(4)]
        try:
            with Octopus(
                model=onnx_path,
                eval_fn=_ort_eval_fn,
                sharding_strategy="pp",
                gpu_ids=[0, 1],
                max_workers=1,
                safety_net_gb=1.0,
                log_level="WARNING",
            ) as o:
                outputs = o.map(inputs)
                assert o.pool_plan is not None
                assert o.pool_plan.sharding_required is True
                assert o.pool_plan.gpus_per_shard == 2
                assert o.pool_plan.total_workers == 1
        finally:
            _cleanup_onnx_export(onnx_path)

        assert len(outputs) == len(inputs)
        assert all(isinstance(out, np.ndarray) for out in outputs)
