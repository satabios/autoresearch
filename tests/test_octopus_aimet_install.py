"""Verify octopus_aimet monkey-patches aimet_onnx.QuantizationSimModel on import.

No GPU required. All aimet_onnx imports are mocked via sys.modules.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_aimet_modules():
    """Return a context manager that injects minimal aimet_onnx stubs."""
    mock_qs_module = MagicMock()
    mock_aimet_onnx = MagicMock()
    mock_aimet_onnx.__version__ = "1.30.0"

    # Provide a concrete class so isinstance checks work later
    class _FakeOriginalQSim:
        def __init__(self, model, *args, **kwargs):
            pass

        def compute_encodings(self, cb, args):
            pass

        def sensitivity_scan(self, eval_fn, layers):
            return {}

    mock_qs_module.QuantizationSimModel = _FakeOriginalQSim
    mock_aimet_onnx.quantsim = mock_qs_module

    mods = {
        "aimet_onnx": mock_aimet_onnx,
        "aimet_onnx.quantsim": mock_qs_module,
        "aimet_onnx.common": MagicMock(),
        "aimet_onnx.common.defs": MagicMock(),
    }
    return patch.dict("sys.modules", mods), mock_qs_module, _FakeOriginalQSim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInstallPatch:
    def test_patch_replaces_quantsim_class(self):
        """After `import octopus_aimet`, aimet_onnx.quantsim.QuantizationSimModel
        should be OctopusQuantSimModel, not the original."""
        ctx, mock_qs_module, _FakeOriginal = _mock_aimet_modules()
        with ctx:
            # Remove any cached octopus_aimet modules so _install() re-runs
            for key in list(sys.modules):
                if key.startswith("octopus_aimet"):
                    del sys.modules[key]

            import octopus_aimet  # triggers _install()

            # The module attribute should now be OctopusQuantSimModel
            patched_cls = mock_qs_module.QuantizationSimModel
            assert patched_cls.__name__ == "OctopusQuantSimModel"

    def test_octopus_quantsim_is_subclass_of_original(self):
        """OctopusQuantSimModel must be a subclass of _Original for isinstance checks."""
        ctx, mock_qs_module, _FakeOriginal = _mock_aimet_modules()
        with ctx:
            for key in list(sys.modules):
                if key.startswith("octopus_aimet"):
                    del sys.modules[key]

            import octopus_aimet

            patched_cls = mock_qs_module.QuantizationSimModel
            assert issubclass(patched_cls, _FakeOriginal)

    def test_original_saved_before_patch(self):
        """octopus_aimet._OriginalQuantSimModel must reference the pre-patch class."""
        ctx, mock_qs_module, _FakeOriginal = _mock_aimet_modules()
        with ctx:
            for key in list(sys.modules):
                if key.startswith("octopus_aimet"):
                    del sys.modules[key]

            import octopus_aimet

            assert octopus_aimet._OriginalQuantSimModel is _FakeOriginal

    def test_import_silently_skips_if_aimet_not_installed(self):
        """If aimet_onnx is absent, octopus_aimet should import without error."""
        # Remove aimet_onnx from sys.modules and block it
        clean_mods = {k: v for k, v in sys.modules.items()
                      if not k.startswith("aimet_onnx")}
        # Use ImportError-raising stub
        class _NoAimet:
            @staticmethod
            def find_module(name, path=None):
                if "aimet_onnx" in name:
                    return _NoAimet
            @staticmethod
            def load_module(name):
                raise ImportError(f"Mocked absence: {name}")

        import importlib
        finder = _NoAimet()

        for key in list(sys.modules):
            if key.startswith("octopus_aimet") or key.startswith("aimet_onnx"):
                del sys.modules[key]

        sys.meta_path.insert(0, finder)
        try:
            import octopus_aimet  # must not raise
        finally:
            sys.meta_path.remove(finder)

    def test_version_warning_when_aimet_too_old(self):
        """octopus_aimet should warn if aimet_onnx < 1.30."""
        ctx, mock_qs_module, _FakeOriginal = _mock_aimet_modules()
        with ctx:
            # Downgrade mock version
            sys.modules["aimet_onnx"].__version__ = "1.29.0"

            for key in list(sys.modules):
                if key.startswith("octopus_aimet"):
                    del sys.modules[key]

            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import octopus_aimet  # noqa: F401

            version_warnings = [x for x in w if "aimet" in str(x.message).lower()
                                 or "1.30" in str(x.message)]
            assert len(version_warnings) >= 1
