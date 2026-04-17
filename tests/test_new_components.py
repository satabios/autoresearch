"""Tests for profile_ort_vram, WorkerOOMError, detect_oom_in_logs, poll_gpu_memory, Ray temp dir, pool MPS/stagger."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from octopus._types import VRAMProfile
from octopus.exceptions import WorkerOOMError, detect_oom_in_logs


# ---------------------------------------------------------------------------
# WorkerOOMError
# ---------------------------------------------------------------------------

class TestWorkerOOMError:
    def test_suggested_sn_is_current_plus_one(self):
        err = WorkerOOMError("OOM", current_sn=2.5)
        assert err.suggested_sn_gb == pytest.approx(3.5)

    def test_default_current_sn_zero(self):
        err = WorkerOOMError("OOM")
        assert err.suggested_sn_gb == pytest.approx(1.0)

    def test_message_contains_suggestion(self):
        err = WorkerOOMError("OOM hit", current_sn=1.0)
        assert "2.0" in str(err)

    def test_worker_id_and_gpu_id_in_message(self):
        err = WorkerOOMError("OOM", worker_id=3, gpu_id=1, current_sn=0.0)
        assert "worker=3" in str(err)
        assert "gpu=1" in str(err)

    def test_oom_pattern_stored(self):
        err = WorkerOOMError("OOM", oom_pattern="BFCArena")
        assert err.oom_pattern == "BFCArena"

    def test_oom_patterns_list_contains_expected(self):
        patterns = WorkerOOMError.OOM_PATTERNS
        assert "BFCArena" in patterns
        assert "CUBLAS_STATUS_ALLOC_FAILED" in patterns
        assert "CUDA out of memory" in patterns


# ---------------------------------------------------------------------------
# detect_oom_in_logs
# ---------------------------------------------------------------------------

class TestDetectOomInLogs:
    def test_detects_bfcarena(self):
        assert detect_oom_in_logs("Error: BFCArena allocation failed") == "BFCArena"

    def test_detects_cublas(self):
        assert detect_oom_in_logs("CUBLAS_STATUS_ALLOC_FAILED in some call") == "CUBLAS_STATUS_ALLOC_FAILED"

    def test_detects_cuda_oom(self):
        # "out of memory" appears before "CUDA out of memory" in OOM_PATTERNS — first match wins
        result = detect_oom_in_logs("RuntimeError: CUDA out of memory")
        assert result in WorkerOOMError.OOM_PATTERNS
        assert result != ""

    def test_returns_empty_for_no_match(self):
        assert detect_oom_in_logs("Everything is fine") == ""

    def test_returns_empty_for_empty_string(self):
        assert detect_oom_in_logs("") == ""


# ---------------------------------------------------------------------------
# profile_ort_vram
# ---------------------------------------------------------------------------

class TestProfileOrtVram:
    def test_fallback_when_cloudpickle_missing(self):
        """Without cloudpickle, should return 4.0 GB fallback."""
        from octopus.profiler import _VRAM_PROBE_FALLBACK_GB

        import sys
        # Remove cloudpickle from sys.modules so the import inside profile_ort_vram raises
        original = sys.modules.pop("cloudpickle", _SENTINEL := object())

        try:
            # Also ensure it can't be imported
            with patch.dict("sys.modules", {"cloudpickle": None}):
                from octopus.profiler import profile_ort_vram
                profile = profile_ort_vram(
                    onnx_path="/fake/model.onnx",
                    sample_feed_fn=lambda: {},
                    gpu_id=0,
                )
            assert profile.peak_vram_gb == pytest.approx(_VRAM_PROBE_FALLBACK_GB)
        finally:
            if original is not _SENTINEL:
                sys.modules["cloudpickle"] = original

    def test_fallback_when_subprocess_fails(self):
        """Subprocess failure should return 4.0 GB fallback."""
        from octopus.profiler import profile_ort_vram, _VRAM_PROBE_FALLBACK_GB

        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps.return_value = b"fake_fn"

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            with patch("octopus.profiler.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1,
                    stderr="subprocess crashed",
                    stdout="",
                )
                profile = profile_ort_vram(
                    onnx_path="/fake/model.onnx",
                    sample_feed_fn=lambda: {},
                    gpu_id=0,
                )

        assert profile.peak_vram_gb == pytest.approx(_VRAM_PROBE_FALLBACK_GB)
        assert profile.profiled_on_device == 0

    def test_success_path_applies_multiplier(self):
        """Successful probe should apply 1.25x multiplier to delta."""
        from octopus.profiler import profile_ort_vram, _VRAM_PROBE_SAFETY_MULTIPLIER

        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps.return_value = b"fake_fn"

        probe_output = json.dumps({
            "baseline_mb": 10000.0,
            "after_load_mb": 8000.0,
            "peak_mb": 7000.0,
            "delta_mb": 3000.0,  # 10000 - 7000
        })

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            with patch("octopus.profiler.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=probe_output + "\n",
                    stderr="",
                )
                profile = profile_ort_vram(
                    onnx_path="/fake/model.onnx",
                    sample_feed_fn=lambda: {},
                    gpu_id=2,
                )

        expected_gb = 3000.0 / 1024.0 * _VRAM_PROBE_SAFETY_MULTIPLIER
        assert profile.peak_vram_gb == pytest.approx(expected_gb, rel=1e-4)
        assert profile.profiled_on_device == 2

    def test_success_path_floor_at_100mb(self):
        """Very small delta should be floored at 0.1 GB."""
        from octopus.profiler import profile_ort_vram

        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps.return_value = b"fake_fn"

        probe_output = json.dumps({
            "baseline_mb": 1000.0,
            "after_load_mb": 999.0,
            "peak_mb": 999.5,
            "delta_mb": 0.5,  # tiny
        })

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            with patch("octopus.profiler.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=probe_output + "\n",
                    stderr="",
                )
                profile = profile_ort_vram(
                    onnx_path="/fake/model.onnx",
                    sample_feed_fn=lambda: {},
                    gpu_id=0,
                )

        assert profile.peak_vram_gb >= 0.1

    def test_returns_vram_profile_type(self):
        from octopus.profiler import profile_ort_vram

        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps.return_value = b"fake_fn"

        probe_output = json.dumps({
            "baseline_mb": 5000.0,
            "after_load_mb": 3000.0,
            "peak_mb": 2500.0,
            "delta_mb": 2500.0,
        })

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            with patch("octopus.profiler.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=probe_output + "\n",
                    stderr="",
                )
                profile = profile_ort_vram("/fake.onnx", lambda: {}, gpu_id=0)

        assert isinstance(profile, VRAMProfile)


# ---------------------------------------------------------------------------
# poll_gpu_memory
# ---------------------------------------------------------------------------

class TestPollGpuMemory:
    def test_returns_free_gb_per_gpu(self):
        from octopus.discovery import poll_gpu_memory

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
            free=int(4.0 * 1024 ** 3)
        )

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = poll_gpu_memory([0, 1])

        assert 0 in result
        assert 1 in result
        assert result[0] == pytest.approx(4.0, rel=1e-3)

    def test_returns_empty_when_pynvml_missing(self):
        from octopus.discovery import poll_gpu_memory

        import sys
        original = sys.modules.get("pynvml")
        sys.modules["pynvml"] = None  # type: ignore

        try:
            result = poll_gpu_memory([0])
            assert result == {}
        finally:
            if original is None:
                del sys.modules["pynvml"]
            else:
                sys.modules["pynvml"] = original


# ---------------------------------------------------------------------------
# ray_runtime
# ---------------------------------------------------------------------------

class TestRayRuntime:
    def test_resolve_ray_temp_dir_prefers_explicit_env(self, tmp_path, monkeypatch):
        from octopus.ray_runtime import resolve_ray_temp_dir

        explicit = tmp_path / "explicit-ray"
        monkeypatch.setenv("OCTOPUS_RAY_TMPDIR", str(explicit))
        monkeypatch.delenv("RAY_TMPDIR", raising=False)

        with patch("octopus.ray_runtime._socket_path_is_safe", return_value=True):
            assert resolve_ray_temp_dir() == str(explicit)

    def test_resolve_ray_temp_dir_picks_path_with_most_free_space(self, tmp_path, monkeypatch):
        from octopus.ray_runtime import resolve_ray_temp_dir

        first = tmp_path / "first"
        second = tmp_path / "second"
        monkeypatch.delenv("OCTOPUS_RAY_TMPDIR", raising=False)
        monkeypatch.delenv("RAY_TMPDIR", raising=False)

        free_bytes = {
            first: 10,
            second: 20,
        }

        with patch("octopus.ray_runtime._DEFAULT_RAY_TEMP_DIRS", (str(first), str(second))), \
             patch("octopus.ray_runtime._socket_path_is_safe", return_value=True), \
             patch("octopus.ray_runtime._path_free_bytes", side_effect=lambda path: free_bytes[Path(path)]):
            assert resolve_ray_temp_dir() == str(second)

    def test_resolve_ray_temp_dir_skips_socket_too_long(self, tmp_path, monkeypatch):
        from octopus.ray_runtime import resolve_ray_temp_dir

        too_long = tmp_path / ("x" * 90)
        short = tmp_path / "short"
        monkeypatch.delenv("OCTOPUS_RAY_TMPDIR", raising=False)
        monkeypatch.delenv("RAY_TMPDIR", raising=False)

        with patch("octopus.ray_runtime._DEFAULT_RAY_TEMP_DIRS", (str(too_long), str(short))), \
             patch(
                 "octopus.ray_runtime._socket_path_is_safe",
                 side_effect=lambda path: Path(path) == short,
             ):
            assert resolve_ray_temp_dir() == str(short)

    def test_build_ray_init_kwargs_adds_temp_dir_only_for_local(self):
        from octopus.ray_runtime import build_ray_init_kwargs

        with patch("octopus.ray_runtime.resolve_ray_temp_dir", return_value="/safe/ray"):
            local = build_ray_init_kwargs(log_to_driver=False)
            remote = build_ray_init_kwargs(address="auto", log_to_driver=False)

        assert local["_temp_dir"] == "/safe/ray"
        assert local["log_to_driver"] is False
        assert "_temp_dir" not in remote
        assert remote["address"] == "auto"

    def test_skips_failed_gpu(self):
        from octopus.discovery import poll_gpu_memory

        mock_pynvml = MagicMock()

        def get_handle(idx):
            if idx == 1:
                raise RuntimeError("GPU 1 error")
            return MagicMock()

        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
            free=int(8.0 * 1024 ** 3)
        )

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = poll_gpu_memory([0, 1])

        assert 0 in result
        assert 1 not in result


# ---------------------------------------------------------------------------
# Pool: MPS env vars
# ---------------------------------------------------------------------------

class TestPoolMpsEnvVars:
    def test_mps_env_vars_injected_when_enabled(self):
        """When enable_mps=True, MPS env vars should be in actor runtime_env."""
        from octopus._types import PoolPlan, WorkerAllocation
        from octopus.pool import WorkerPool

        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=2, num_workers=1,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
            ],
            total_workers=1,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        captured_env_vars = []

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.InferenceWorker") as MockWorker:

            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "ref"
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            MockWorker.options.return_value = mock_options
            mock_ray.get.return_value = [{"status": "ready"}]

            def capture_options(**kwargs):
                env = kwargs.get("runtime_env", {}).get("env_vars", {})
                captured_env_vars.append(env)
                return mock_options

            MockWorker.options.side_effect = capture_options

            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
                enable_mps=True,
            )
            pool.start()

        assert len(captured_env_vars) == 1
        env = captured_env_vars[0]
        assert "CUDA_MPS_PIPE_DIRECTORY" in env
        assert "CUDA_MPS_LOG_DIRECTORY" in env
        assert "gpu2" in env["CUDA_MPS_PIPE_DIRECTORY"]

    def test_no_mps_env_vars_when_disabled(self):
        from octopus._types import PoolPlan, WorkerAllocation
        from octopus.pool import WorkerPool

        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=0, num_workers=1,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
            ],
            total_workers=1,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        captured_env_vars = []

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.InferenceWorker") as MockWorker:

            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "ref"
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            mock_ray.get.return_value = [{"status": "ready"}]

            def capture_options(**kwargs):
                env = kwargs.get("runtime_env", {}).get("env_vars", {})
                captured_env_vars.append(env)
                return mock_options

            MockWorker.options.side_effect = capture_options

            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
                enable_mps=False,
            )
            pool.start()

        env = captured_env_vars[0]
        assert "CUDA_MPS_PIPE_DIRECTORY" not in env
        assert "CUDA_MPS_LOG_DIRECTORY" not in env


# ---------------------------------------------------------------------------
# Pool: CUDA init stagger
# ---------------------------------------------------------------------------

class TestPoolStagger:
    def test_stagger_sleeps_between_slots(self):
        """With stagger_init_s > 0, time.sleep should be called between slots."""
        from octopus._types import PoolPlan, WorkerAllocation
        from octopus.pool import WorkerPool

        # 2 workers on same GPU → 1 sleep between slot 0 and slot 1
        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=0, num_workers=2,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
            ],
            total_workers=2,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        sleep_calls = []

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.InferenceWorker") as MockWorker, \
             patch("octopus.pool.time.sleep", side_effect=lambda s: sleep_calls.append(s)):

            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "ref"
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            MockWorker.options.return_value = mock_options
            mock_ray.get.return_value = [{"status": "ready"}, {"status": "ready"}]

            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
                stagger_init_s=5.0,
            )
            pool.start()

        # 2 workers on 1 GPU → 1 sleep (between slot 0 and slot 1)
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(5.0)

    def test_no_stagger_no_sleep(self):
        from octopus._types import PoolPlan, WorkerAllocation
        from octopus.pool import WorkerPool

        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=0, num_workers=2,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
            ],
            total_workers=2,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        sleep_calls = []

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.InferenceWorker") as MockWorker, \
             patch("octopus.pool.time.sleep", side_effect=lambda s: sleep_calls.append(s)):

            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "ref"
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            MockWorker.options.return_value = mock_options
            mock_ray.get.return_value = [{"status": "ready"}, {"status": "ready"}]

            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
                stagger_init_s=0.0,
            )
            pool.start()

        assert sleep_calls == []

    def test_different_gpu_workers_not_staggered(self):
        """Workers on different GPUs should all init in parallel (no sleep)."""
        from octopus._types import PoolPlan, WorkerAllocation
        from octopus.pool import WorkerPool

        plan = PoolPlan(
            allocations=[
                WorkerAllocation(device_id=0, num_workers=1,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
                WorkerAllocation(device_id=1, num_workers=1,
                                 vram_per_worker_gb=5.0, reserved_safety_gb=1.0),
            ],
            total_workers=2,
            sharding_required=False,
            sharding_strategy=None,
            gpus_per_shard=1,
        )

        sleep_calls = []

        with patch("octopus.pool.ray") as mock_ray, \
             patch("octopus.pool.InferenceWorker") as MockWorker, \
             patch("octopus.pool.time.sleep", side_effect=lambda s: sleep_calls.append(s)):

            mock_actor = MagicMock()
            mock_actor.initialize.remote.return_value = "ref"
            mock_options = MagicMock()
            mock_options.remote.return_value = mock_actor
            MockWorker.options.return_value = mock_options
            mock_ray.get.return_value = [{"status": "ready"}, {"status": "ready"}]

            pool = WorkerPool(
                pool_plan=plan,
                model_bytes=b"fake",
                adapter_cls_name="pytorch",
                eval_fn=lambda m, b: None,
                stagger_init_s=5.0,
            )
            pool.start()

        # 1 worker per GPU, max_per_gpu=1 → only 1 slot → no sleep
        assert sleep_calls == []
