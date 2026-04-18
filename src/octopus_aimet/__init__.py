from __future__ import annotations

import warnings

_AIMET_MIN = (1, 30)

# Original QuantizationSimModel saved before any monkey-patching.
# _calibration.py uses this to construct fresh QuantSim on workers
# without triggering OctopusQuantSimModel.__init__ recursion.
_OriginalQuantSimModel = None


def _install() -> None:
    global _OriginalQuantSimModel
    try:
        import aimet_onnx.quantsim as _qs_module  # type: ignore[import-untyped]
    except ImportError:
        return  # aimet_onnx not installed; silently skip

    # Version guard — biweekly AIMET releases can break internal APIs
    try:
        import aimet_onnx  # type: ignore[import-untyped]

        ver = tuple(int(x) for x in aimet_onnx.__version__.split(".")[:2])
        if ver < _AIMET_MIN:
            warnings.warn(
                f"octopus_aimet tested with aimet-onnx>={_AIMET_MIN}; got {ver}. "
                "Some APIs may be incompatible.",
                UserWarning,
                stacklevel=2,
            )
    except Exception:
        pass

    from octopus_aimet._patch import OctopusQuantSimModel, _Original

    _OriginalQuantSimModel = _Original
    _qs_module.QuantizationSimModel = OctopusQuantSimModel


_install()
