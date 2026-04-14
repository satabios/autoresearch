from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn

from octopus._types import GPUInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_octopus_with_mock_pool(results_per_submit=None):
    """Return an Octopus instance whose pool is fully mocked.

    ``results_per_submit`` is a list of values that pool.submit() will return
    (as fake ObjectRefs).  ray.get() is also mocked to return them directly.
    """
    from octopus.core import Octopus

    model = nn.Linear(10, 5)
    o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")

    mock_pool = MagicMock()
    if results_per_submit is not None:
        # pool.submit() returns a fake ref; ray.get() resolves refs → values
        fake_refs = [MagicMock(name=f"ref_{i}") for i in range(len(results_per_submit))]
        mock_pool.submit.side_effect = fake_refs
    else:
        fake_refs = []

    o._pool = mock_pool
    return o, mock_pool, fake_refs, results_per_submit or []


class TestOctopusCore:
    """Unit tests for the Octopus orchestrator (mocked GPU/Ray)."""

    def test_init_detects_adapter(self):
        """Octopus should detect the model type on construction."""
        with patch("octopus.core.discover_gpus"), \
             patch("octopus.core.ray"):
            from octopus.core import Octopus

            model = nn.Linear(10, 5)
            o = Octopus(
                model=model,
                eval_fn=lambda m, b: m(b),
                log_level="WARNING",
            )
            assert o._adapter.model_type_name == "pytorch"
            assert o.num_workers == 0  # not initialized yet
            assert o.gpu_info == []

    def test_context_manager_calls_shutdown(self):
        """Exiting context should call shutdown."""
        with patch("octopus.core.discover_gpus"), \
             patch("octopus.core.ray") as mock_ray:
            mock_ray.is_initialized.return_value = False
            from octopus.core import Octopus

            model = nn.Linear(10, 5)
            o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")
            mock_pool = MagicMock()
            o._pool = mock_pool

            with o:
                pass

            # shutdown() was called, which calls pool.shutdown()
            mock_pool.shutdown.assert_called_once()

    def test_properties_before_init(self):
        """Properties should return defaults before first call."""
        with patch("octopus.core.discover_gpus"), \
             patch("octopus.core.ray"):
            from octopus.core import Octopus

            model = nn.Linear(10, 5)
            o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")

            assert o.num_workers == 0
            assert o.gpu_info == []
            assert o.vram_profile is None
            assert o.pool_plan is None

    def test_unsupported_model_raises(self):
        """Passing an unsupported model type should raise TypeError."""
        with patch("octopus.core.ray"):
            from octopus.core import Octopus

            with pytest.raises(TypeError, match="Unsupported"):
                Octopus(model="not_a_model", eval_fn=lambda m, b: None)


# ---------------------------------------------------------------------------
# submit / gather / map / results
# ---------------------------------------------------------------------------

class TestSubmitGatherMap:
    """Tests for the parallel for-loop API: submit / gather / map / results."""

    def _make_o(self):
        """Octopus with pool already injected (skips GPU discovery/profiling)."""
        from octopus.core import Octopus

        model = nn.Linear(10, 5)
        o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")
        return o

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------

    def test_submit_calls_pool_submit(self):
        """submit() should delegate to pool.submit() and track the ref."""
        o = self._make_o()
        mock_pool = MagicMock()
        fake_ref = MagicMock(name="ref_0")
        mock_pool.submit.return_value = fake_ref
        o._pool = mock_pool

        returned = o.submit("batch_0")

        mock_pool.submit.assert_called_once_with("batch_0")
        assert returned is fake_ref
        assert fake_ref in o._pending_refs

    def test_submit_accumulates_refs(self):
        """Multiple submit() calls should accumulate refs in _pending_refs."""
        o = self._make_o()
        mock_pool = MagicMock()
        refs = [MagicMock(name=f"ref_{i}") for i in range(3)]
        mock_pool.submit.side_effect = refs
        o._pool = mock_pool

        for i in range(3):
            o.submit(f"batch_{i}")

        assert len(o._pending_refs) == 3
        assert o._pending_refs == refs

    def test_submit_triggers_lazy_init_on_first_call(self):
        """submit() should call _ensure_initialized with the batch."""
        o = self._make_o()
        init_calls = []

        def fake_ensure(batch):
            init_calls.append(batch)
            # Inject a mock pool so submit can proceed
            mock_pool = MagicMock()
            mock_pool.submit.return_value = MagicMock()
            o._pool = mock_pool

        o._ensure_initialized = fake_ensure
        o.submit("first_batch")

        assert init_calls == ["first_batch"]

    # ------------------------------------------------------------------
    # gather
    # ------------------------------------------------------------------

    def test_gather_returns_results_in_order(self):
        """gather() should return results in submission order."""
        o = self._make_o()
        mock_pool = MagicMock()
        refs = [MagicMock(name=f"ref_{i}") for i in range(3)]
        mock_pool.submit.side_effect = refs

        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = ["result_0", "result_1", "result_2"]
            o._pool = mock_pool

            for i in range(3):
                o.submit(f"batch_{i}")

            results = o.gather()

        assert results == ["result_0", "result_1", "result_2"]
        mock_ray.get.assert_called_once_with(refs)

    def test_gather_clears_pending_refs(self):
        """After gather(), _pending_refs should be empty."""
        o = self._make_o()
        mock_pool = MagicMock()
        mock_pool.submit.return_value = MagicMock()
        o._pool = mock_pool

        o.submit("batch")

        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = ["result"]
            o.gather()

        assert o._pending_refs == []

    def test_gather_with_no_pending_returns_existing_results(self):
        """gather() with nothing pending should return previously collected results."""
        o = self._make_o()
        o._results = ["old_result"]

        results = o.gather()
        assert results == ["old_result"]

    def test_gather_accumulates_across_cycles(self):
        """Results from multiple submit/gather cycles should accumulate in o.results."""
        o = self._make_o()
        mock_pool = MagicMock()
        mock_pool.submit.return_value = MagicMock()
        o._pool = mock_pool

        # Cycle 1
        o.submit("b0")
        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = ["r0"]
            o.gather()

        # Cycle 2
        o.submit("b1")
        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = ["r1"]
            o.gather()

        assert o.results == ["r0", "r1"]

    # ------------------------------------------------------------------
    # map
    # ------------------------------------------------------------------

    def test_map_submits_all_and_gathers(self):
        """map() should submit every item and return gathered results."""
        o = self._make_o()
        mock_pool = MagicMock()
        refs = [MagicMock(name=f"ref_{i}") for i in range(4)]
        mock_pool.submit.side_effect = refs
        o._pool = mock_pool

        inputs = ["a", "b", "c", "d"]

        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = [10, 20, 30, 40]
            results = o.map(inputs)

        assert results == [10, 20, 30, 40]
        assert mock_pool.submit.call_count == 4

    def test_map_empty_input_returns_empty(self):
        """map([]) should return [] without touching the pool."""
        o = self._make_o()
        mock_pool = MagicMock()
        o._pool = mock_pool

        results = o.map([])
        assert results == []
        mock_pool.submit.assert_not_called()

    # ------------------------------------------------------------------
    # results property
    # ------------------------------------------------------------------

    def test_results_property_empty_before_gather(self):
        """results should be [] before any gather() call."""
        o = self._make_o()
        assert o.results == []

    def test_results_property_after_gather(self):
        """results should reflect what gather() collected."""
        o = self._make_o()
        mock_pool = MagicMock()
        mock_pool.submit.return_value = MagicMock()
        o._pool = mock_pool

        o.submit("batch")
        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = ["answer"]
            o.gather()

        assert o.results == ["answer"]

    def test_results_returns_copy(self):
        """Mutating the returned list should not affect internal state."""
        o = self._make_o()
        o._results = [1, 2, 3]
        r = o.results
        r.append(99)
        assert o.results == [1, 2, 3]

    # ------------------------------------------------------------------
    # __exit__ auto-gather
    # ------------------------------------------------------------------

    def test_exit_auto_gathers_pending_refs(self):
        """__exit__ should drain _pending_refs before shutdown."""
        from octopus.core import Octopus

        model = nn.Linear(10, 5)
        o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")

        mock_pool = MagicMock()
        fake_ref = MagicMock(name="ref_0")
        mock_pool.submit.return_value = fake_ref
        o._pool = mock_pool
        o._pending_refs = [fake_ref]

        with patch("octopus.core.ray") as mock_ray:
            mock_ray.get.return_value = ["auto_result"]
            mock_ray.is_initialized.return_value = False
            o.__exit__(None, None, None)

        assert o._pending_refs == []
        assert "auto_result" in o._results

    def test_exit_no_pending_skips_gather(self):
        """__exit__ with no pending refs should not call ray.get."""
        from octopus.core import Octopus

        model = nn.Linear(10, 5)
        o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")
        mock_pool = MagicMock()
        o._pool = mock_pool

        with patch("octopus.core.ray") as mock_ray:
            mock_ray.is_initialized.return_value = False
            o.__exit__(None, None, None)

        mock_ray.get.assert_not_called()

    def test_context_manager_for_loop_pattern(self):
        """Demonstrate the primary use-case: with Octopus() as o: for i in range(N): o.submit(...)"""
        from octopus.core import Octopus

        model = nn.Linear(10, 5)

        with patch("octopus.core.ray") as mock_ray:
            mock_ray.is_initialized.return_value = False

            o = Octopus(model=model, eval_fn=lambda m, b: m(b), log_level="WARNING")
            mock_pool = MagicMock()
            refs = [MagicMock(name=f"ref_{i}") for i in range(5)]
            mock_pool.submit.side_effect = refs
            o._pool = mock_pool

            mock_ray.get.return_value = [i * 10 for i in range(5)]

            with o:
                for i in range(5):
                    o.submit(f"input_{i}")
                results = o.gather()

        assert results == [0, 10, 20, 30, 40]
        assert mock_pool.submit.call_count == 5
