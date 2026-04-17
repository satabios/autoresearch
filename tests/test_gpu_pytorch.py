"""GPU integration tests — PyTorchAdapter + InferenceWorker.

Goal: pack as many InferenceWorkers onto available GPUs as VRAM allows,
run all workers in parallel, and verify correctness.

Run:
    pytest tests/test_gpu_pytorch.py -v -m gpu
    pytest tests/test_gpu_pytorch.py -v -m gpu -k "not multi_gpu"  # single-GPU CI
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import ray
import torch
import torch.nn as nn

from octopus._types import GPUInfo, VRAMProfile
from octopus.adapters.pytorch import PyTorchAdapter
from octopus.discovery import compute_worker_allocation, discover_gpus
from octopus.ray_runtime import init_ray
from octopus.worker import InferenceWorker, ShardedInferenceWorkerGroup

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """Two-layer MLP: in_dim → hidden_dim → out_dim (no bias)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=False),
    )


def _model_vram_gb(model: nn.Module) -> float:
    """Parameter VRAM estimate (float32 only, no activations)."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1 << 30)


def _make_filling_model(available_gb: float, safety_gb: float = 0.5, k: int = 4) -> nn.Module:
    """Return an MLP sized so that k copies fill (available_gb - safety_gb) of VRAM.

    Model: Linear(H, 4H) + Linear(4H, H)
    Params: H*4H + 4H*H = 8H² (float32) = 32H² bytes
    """
    target_bytes = int((available_gb - safety_gb) / k * (1 << 30))
    H = max(128, int(math.sqrt(max(target_bytes, 0) / 32)))
    return _mlp(H, H * 4, H)


def _eval_fn(model: nn.Module, batch: torch.Tensor) -> float:
    """Eval fn for PyTorchAdapter: sum of output (scalar)."""
    return model(batch).sum().item()


class _PipelineToyModel(nn.Module):
    """Small model with explicit layer stack for pipeline-parallel smoke tests."""

    def __init__(self, width: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(depth)]
        )
        self.head = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return self.head(x)


@pytest.fixture(scope="module")
def ray_session():
    """Initialize Ray once per module; do not shut down (other modules may use it)."""
    _skip_if_no_cuda()
    if not ray.is_initialized():
        init_ray(ignore_reinit_error=True)
    yield


@pytest.fixture(scope="module")
def gpu_infos():
    """Discover real GPUs via torch.cuda; skip if none."""
    _skip_if_no_cuda()
    return discover_gpus(use_pynvml=False)


# ---------------------------------------------------------------------------
# PyTorchAdapter VRAM sanity
# ---------------------------------------------------------------------------

class TestPyTorchAdapterVRAM:
    """Verify PyTorchAdapter occupies expected VRAM when loaded onto a GPU."""

    def test_adapter_vram_within_tolerance(self, gpu_infos):
        """Loaded model should use ≥80% of its parameter size (float32) in VRAM."""
        device_id = gpu_infos[0].device_id
        device = torch.device(f"cuda:{device_id}")

        # ~256 MB parameter model
        model = _mlp(4096, 16384, 4096)
        expected_gb = _model_vram_gb(model)

        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device_id)

        adapter.load_to_device(device)
        after_load = torch.cuda.memory_allocated(device_id)
        allocated_gb = (after_load - baseline) / (1 << 30)

        adapter.unload()
        torch.cuda.empty_cache()

        # Tolerance: at least 80% of param bytes should appear, at most 1.5× (alignment + buffers)
        assert allocated_gb >= expected_gb * 0.8, (
            f"Expected ≥{expected_gb:.3f} GB but got {allocated_gb:.3f} GB"
        )
        assert allocated_gb <= expected_gb * 1.5, (
            f"Unexpected VRAM overhead: {allocated_gb:.3f} GB vs params {expected_gb:.3f} GB"
        )

    def test_adapter_unload_frees_vram(self, gpu_infos):
        """Unload should free the model's VRAM (within 10 MB tolerance)."""
        device_id = gpu_infos[0].device_id
        device = torch.device(f"cuda:{device_id}")

        model = _mlp(2048, 8192, 2048)
        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)

        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(device_id)

        adapter.load_to_device(device)
        adapter.unload()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(device_id)

        leaked_mb = (after - before) / (1 << 20)
        assert leaked_mb < 10, f"Memory leak after unload: {leaked_mb:.1f} MB"


# ---------------------------------------------------------------------------
# InferenceWorker — single GPU, max K workers
# ---------------------------------------------------------------------------

class TestInferenceWorkerMaxK:
    """Pack the GPU with K workers, run all in parallel, verify correctness."""

    def test_max_k_workers_parallel_run(self, ray_session, gpu_infos):
        """Spawn K workers on available GPUs, submit K*3 batches, check results."""
        safety_net_gb = 0.5

        # Size model to pack ~4 copies per GPU
        model = _make_filling_model(
            gpu_infos[0].available_vram_gb,
            safety_gb=safety_net_gb,
            k=4,
        )
        model_vram_gb = _model_vram_gb(model)
        H_in = model[0].in_features

        plan = compute_worker_allocation(
            gpu_infos,
            model_vram_gb=model_vram_gb,
            safety_net_gb=safety_net_gb,
            max_workers=min(max(2, len(gpu_infos) * 2), 8),
        )
        K = plan.total_workers
        assert K >= 1, "Should fit at least 1 worker"

        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)
        model_bytes = adapter.state_bytes()

        workers: list = []
        for alloc in plan.allocations:
            frac = 1.0 / alloc.num_workers
            for _ in range(alloc.num_workers):
                w = InferenceWorker.options(
                    num_gpus=frac,
                ).remote(model_bytes, "pytorch", _eval_fn)
                workers.append(w)

        try:
            # Initialize all workers in parallel
            init_refs = [w.initialize.remote() for w in workers]
            init_results = ray.get(init_refs, timeout=180)

            assert len(init_results) == K
            for r in init_results:
                assert r["status"] == "ready"
                assert r["memory_allocated_mb"] > 0

            # Submit K*3 batches round-robin across all workers simultaneously
            batches = [torch.randn(2, H_in) for _ in range(K * 3)]
            run_refs = [
                workers[i % K].run.remote(batches[i])
                for i in range(len(batches))
            ]
            results = ray.get(run_refs, timeout=120)

            assert len(results) == K * 3
            for r in results:
                assert isinstance(r, float), f"Expected float, got {type(r)}"
                assert math.isfinite(r), "Result should be finite"

        finally:
            shutdown_refs = [w.shutdown.remote() for w in workers]
            try:
                ray.get(shutdown_refs, timeout=30)
            except Exception:
                pass
            for w in workers:
                try:
                    ray.kill(w)
                except Exception:
                    pass

    def test_worker_memory_reported_nonzero(self, ray_session, gpu_infos):
        """initialize() must report >0 MB allocated (confirms GPU placement)."""
        model = _mlp(2048, 8192, 2048)
        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)
        model_bytes = adapter.state_bytes()

        w = InferenceWorker.options(
            num_gpus=0.1,
        ).remote(model_bytes, "pytorch", _eval_fn)

        try:
            result = ray.get(w.initialize.remote(), timeout=60)
            assert result["status"] == "ready"
            assert result["memory_allocated_mb"] > 0
        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=15)
            except Exception:
                pass
            ray.kill(w)

    def test_health_check_true_after_init(self, ray_session, gpu_infos):
        """health_check() should return True after successful initialize()."""
        model = nn.Linear(1024, 1024, bias=False)
        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)
        model_bytes = adapter.state_bytes()

        w = InferenceWorker.options(
            num_gpus=0.1,
        ).remote(model_bytes, "pytorch", _eval_fn)

        try:
            ray.get(w.initialize.remote(), timeout=60)
            assert ray.get(w.health_check.remote(), timeout=15) is True
        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=15)
            except Exception:
                pass
            ray.kill(w)

    def test_worker_run_output_numerically_correct(self, ray_session, gpu_infos):
        """Worker output should match CPU forward pass to within float32 tolerance."""
        model = nn.Linear(256, 256, bias=False)
        torch.manual_seed(42)
        nn.init.xavier_uniform_(model.weight)

        # Known input → compute expected output on CPU
        batch = torch.randn(1, 256, generator=torch.Generator().manual_seed(7))
        expected = model(batch).sum().item()

        def exact_eval(m: nn.Module, b: torch.Tensor) -> float:
            return m(b).sum().item()

        adapter = PyTorchAdapter(model, eval_fn=exact_eval)
        model_bytes = adapter.state_bytes()

        w = InferenceWorker.options(
            num_gpus=0.1,
        ).remote(model_bytes, "pytorch", exact_eval)

        try:
            ray.get(w.initialize.remote(), timeout=60)
            result = ray.get(w.run.remote(batch), timeout=30)
            assert abs(result - expected) < 1e-4, (
                f"GPU result {result} differs from CPU expected {expected}"
            )
        finally:
            try:
                ray.get(w.shutdown.remote(), timeout=15)
            except Exception:
                pass
            ray.kill(w)


# ---------------------------------------------------------------------------
# Multi-GPU: workers spread across all available GPUs
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestPyTorchMultiGPU:
    """Verify workers are distributed across all available GPUs."""

    def test_workers_span_all_gpus(self, ray_session, gpu_infos):
        if len(gpu_infos) < 2:
            pytest.skip("Requires ≥2 GPUs")

        model = _mlp(2048, 8192, 2048)
        model_vram_gb = _model_vram_gb(model)
        H_in = model[0].in_features

        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)
        model_bytes = adapter.state_bytes()

        plan = compute_worker_allocation(
            gpu_infos,
            model_vram_gb,
            safety_net_gb=0.5,
            max_workers=max(2, len(gpu_infos)),
        )
        assert len(plan.allocations) > 1, (
            "With ≥2 GPUs and small model, workers should span multiple GPUs"
        )

        workers: list = []
        for alloc in plan.allocations:
            frac = 1.0 / alloc.num_workers
            for _ in range(alloc.num_workers):
                w = InferenceWorker.options(
                    num_gpus=frac,
                ).remote(model_bytes, "pytorch", _eval_fn)
                workers.append(w)

        try:
            init_refs = [w.initialize.remote() for w in workers]
            results = ray.get(init_refs, timeout=180)
            assert all(r["status"] == "ready" for r in results)
            assert all(r["memory_allocated_mb"] > 0 for r in results)

            # Run one batch per worker in parallel
            K = len(workers)
            batches = [torch.randn(2, H_in) for _ in range(K)]
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
class TestPyTorchShardedPipelineParallel:
    """Real shared-model path on multiple GPUs via pipeline parallel."""

    def test_sharded_worker_group_pipeline_parallel_run(self, ray_session):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >=2 GPUs")

        model = _PipelineToyModel(width=128, depth=4)
        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)
        model_bytes = adapter.state_bytes()
        batch = torch.randn(2, 128)

        worker = ShardedInferenceWorkerGroup.options(num_gpus=2).remote(
            model_bytes,
            "pytorch",
            _eval_fn,
            [0, 1],
            "pp",
            0,
        )

        try:
            init_result = ray.get(worker.initialize.remote(), timeout=120)
            assert init_result["status"] == "ready"
            assert init_result["strategy"] == "pp"
            assert init_result["visible_gpu_count"] == 2
            assert len(init_result["assigned_gpu_ids"]) == 2

            result = ray.get(worker.run.remote(batch), timeout=60)
            assert isinstance(result, float)
            assert math.isfinite(result)
        finally:
            try:
                ray.get(worker.shutdown.remote(), timeout=30)
            except Exception:
                pass
            ray.kill(worker)

    def test_octopus_pipeline_parallel_map_with_two_groups(self, ray_session, monkeypatch):
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires >=4 GPUs")

        from octopus.core import Octopus

        fake_gpus = [
            GPUInfo(
                device_id=i,
                name=f"GPU-{i}",
                total_vram_gb=80.0,
                available_vram_gb=70.0,
                compute_capability=(8, 0),
            )
            for i in range(4)
        ]
        fake_profile = VRAMProfile(
            peak_vram_bytes=0,
            peak_vram_gb=90.0,
            model_params_bytes=0,
            activation_peak_bytes=0,
            profiled_on_device=0,
        )

        monkeypatch.setattr("octopus.core.discover_gpus", lambda device_ids=None: fake_gpus)
        monkeypatch.setattr(
            "octopus.core.profile_model_vram",
            lambda adapter, sample_batch, device_id: fake_profile,
        )

        model = _PipelineToyModel(width=128, depth=4)
        inputs = [torch.randn(2, 128) for _ in range(6)]

        with Octopus(
            model=model,
            eval_fn=_eval_fn,
            sharding_strategy="pp",
            gpu_ids=[0, 1, 2, 3],
            max_workers=2,
            safety_net_gb=1.0,
            log_level="WARNING",
        ) as o:
            results = o.map(inputs)
            assert o.pool_plan is not None
            assert o.pool_plan.sharding_required is True
            assert o.pool_plan.gpus_per_shard == 2
            assert o.pool_plan.total_workers == 2

        assert len(results) == len(inputs)
        assert all(isinstance(r, float) and math.isfinite(r) for r in results)


@pytest.mark.gpu
class TestPyTorchShardedTensorParallel:
    """Real shared-model path on multiple GPUs via tensor parallel."""

    def test_sharded_worker_group_tensor_parallel_run(self, ray_session):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >=2 GPUs")

        model = _mlp(128, 256, 128)
        adapter = PyTorchAdapter(model, eval_fn=_eval_fn)
        model_bytes = adapter.state_bytes()
        batch = torch.randn(2, 128)

        worker = ShardedInferenceWorkerGroup.options(num_gpus=2).remote(
            model_bytes,
            "pytorch",
            _eval_fn,
            [0, 1],
            "tp",
            0,
        )

        try:
            init_result = ray.get(worker.initialize.remote(), timeout=120)
            assert init_result["status"] == "ready"
            assert init_result["strategy"] == "tp"
            assert init_result["visible_gpu_count"] == 2
            assert len(init_result["assigned_gpu_ids"]) == 2

            result = ray.get(worker.run.remote(batch), timeout=60)
            assert isinstance(result, float)
            assert math.isfinite(result)
        finally:
            try:
                ray.get(worker.shutdown.remote(), timeout=30)
            except Exception:
                pass
            ray.kill(worker)

    def test_octopus_tensor_parallel_map(self, ray_session, monkeypatch):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >=2 GPUs")

        from octopus.core import Octopus

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
        monkeypatch.setattr(
            "octopus.core.profile_model_vram",
            lambda adapter, sample_batch, device_id: fake_profile,
        )

        model = _mlp(128, 256, 128)
        inputs = [torch.randn(2, 128) for _ in range(6)]

        with Octopus(
            model=model,
            eval_fn=_eval_fn,
            sharding_strategy="tp",
            gpu_ids=[0, 1],
            max_workers=1,
            safety_net_gb=1.0,
            log_level="WARNING",
        ) as o:
            results = o.map(inputs)
            assert o.pool_plan is not None
            assert o.pool_plan.sharding_required is True
            assert o.pool_plan.gpus_per_shard == 2
            assert o.pool_plan.total_workers == 1

        assert len(results) == len(inputs)
        assert all(isinstance(r, float) and math.isfinite(r) for r in results)


# ---------------------------------------------------------------------------
# Full Octopus API — context manager + for-loop pattern
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestOctopusForLoopGPU:
    """End-to-end test of the Octopus context manager on a real GPU."""

    def test_octopus_submit_gather_on_gpu(self, ray_session, gpu_infos):
        """with Octopus(model, eval_fn) as o: for i in ...: o.submit(inputs[i])."""
        from octopus.core import Octopus

        H = 512
        model = nn.Linear(H, H, bias=False)

        def eval_fn(m: nn.Module, b: torch.Tensor) -> float:
            return m(b).sum().item()

        N = 8
        inputs = [torch.randn(2, H) for _ in range(N)]

        with Octopus(
            model=model,
            eval_fn=eval_fn,
            safety_net_gb=0.5,
            workers_per_gpu=1,
            log_level="WARNING",
        ) as o:
            for inp in inputs:
                o.submit(inp)
            results = o.gather()

        assert len(results) == N
        for r in results:
            assert isinstance(r, float)
            assert math.isfinite(r)

    def test_octopus_map_on_gpu(self, ray_session, gpu_infos):
        """o.map(dataset) should process all items and return ordered results."""
        from octopus.core import Octopus

        H = 256
        model = nn.Linear(H, H, bias=False)

        def eval_fn(m: nn.Module, b: torch.Tensor) -> float:
            return m(b).mean().item()

        N = 6
        dataset = [torch.randn(1, H) for _ in range(N)]

        with Octopus(
            model=model,
            eval_fn=eval_fn,
            workers_per_gpu=1,
            log_level="WARNING",
        ) as o:
            results = o.map(dataset)

        assert len(results) == N
        assert all(math.isfinite(r) for r in results)
