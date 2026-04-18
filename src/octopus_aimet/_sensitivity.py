from __future__ import annotations

from typing import Callable


def run_sensitivity_scan(
    sim: object,
    eval_fn: Callable,
    layers: list[str],
) -> dict[str, float]:
    """Run distributed per-layer SQNR sensitivity analysis.

    Constructs an Octopus instance backed by OnnxQuantSimAdapter,
    then delegates to Octopus.sensitivity_scan() which spawns
    SensitivityWorker Ray actors and distributes the enabling loop.

    Args:
        sim: aimet_onnx QuantizationSimModel (may be OctopusQuantSimModel).
        eval_fn: Callable(ort.InferenceSession) -> float.
                 Called once per layer with all quantizers disabled except that layer.
        layers: op names to evaluate (from sim.qc_quantize_op_dict.keys()).

    Returns:
        {layer_name: sqnr_score}
    """
    from octopus.core import Octopus

    # detect_and_wrap recognises QuantizationSimModel (including subclasses)
    # and wraps in OnnxQuantSimAdapter automatically.
    oct = Octopus(model=sim, eval_fn=eval_fn)
    try:
        return oct.sensitivity_scan(layers=layers)
    finally:
        oct.shutdown()
