from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from octopus._types import GPUInfo


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
