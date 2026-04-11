"""
sensitivity_worker.py — per-layer sensitivity scan worker
==========================================================

Subprocess worker launched by parallel_sensitivity.run_parallel_sensitivity().
Runs the enabling-analysis loop on an assigned subset of layers and writes
{layer_name: sqnr_score} to --output-file.

Usage (internal — called by parallel_sensitivity.py):
    CUDA_VISIBLE_DEVICES=<gpu_id> python sensitivity_worker.py \\
        --onnx-path       <path/to/worker_sim.onnx>       \\
        --encodings-path  <path/to/worker_sim.encodings>  \\
        --module-alias    <alias>                          \\
        --data-path       <data_path>                      \\
        --qscheme-key     <fp16|w8a16|w8a8|w4a16|w4a4>    \\
        --layers-file     <path/to/chunk_i.json>           \\
        --output-file     <path/to/result_i.json>          \\
        --gpu             0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path — must happen before any local imports
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

import aimet_onnx
from aimet_onnx import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.qc_quantize_op import OpMode
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException as OrtRuntimeException
from pipeline.data_loader import create_module_dataloader

os.environ["HF_HOME"] = "./"

# ---------------------------------------------------------------------------
# Logging — all output goes to stdout so the parent process can tee it to a
# per-worker log file via Popen(stdout=log_fh, stderr=STDOUT).
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("sensitivity_worker")

# ---------------------------------------------------------------------------
# QScheme type map
# Mirrors quant.py QSCHEMES — defined here independently to avoid importing
# quant.py, which has module-level side effects (runs the full quant loop).
# ---------------------------------------------------------------------------
QSCHEME_TYPES: dict[str, tuple] = {
    "fp16":  (aimet_onnx.float16, aimet_onnx.float16),
    "w8a16": (aimet_onnx.int8,    aimet_onnx.int16),
    "w8a8":  (aimet_onnx.int8,    aimet_onnx.int8),
    "w4a16": (aimet_onnx.int4,    aimet_onnx.int16),
    "w4a4":  (aimet_onnx.int4,    aimet_onnx.int4),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_names(module_alias: str) -> list[str]:
    """Return the expected output tensor names for a given module alias."""
    return ["route", "speed"] if module_alias in ("llm_ar32", "Qwen2_05_VLM") else ["output"]


def _build_providers(gpu_id: int) -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    log.warning(
        "CUDAExecutionProvider not in ORT available providers %s — falling back to CPU. "
        "Install onnxruntime-gpu and ensure CUDA libs are on LD_LIBRARY_PATH.",
        available,
    )
    return ["CPUExecutionProvider"]


def _to_numpy(obj: Any) -> Any:
    """Recursively convert torch tensors to numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_numpy(x) for x in obj]
    return obj


def _make_feed(session: ort.InferenceSession, inputs: Any, module_alias: str) -> dict:
    """Build the ORT input feed dict.

    Mirrors Quanter._make_feed / MODULE_ORT_INPUT_ALIAS logic from quant.py:
      • InternViT alias  → single-input passthrough (fused ONNX has one input)
      • Qwen2_05_VLM     → named multi-input mapping, drop input_ids (index 2)
      • everything else  → positional zip against session input names
    """
    input_names = [inp.name for inp in session.get_inputs()]

    # Fused single-input modules (InternViT300M_Pixel_Unshuffle_MLP1)
    if module_alias == "InternViT":
        return {input_names[0]: inputs}

    # Multi-input modules
    if isinstance(inputs, (list, tuple)):
        # Qwen2_05_VLM: drop input_ids at index 2, map the rest by name
        if module_alias == "Qwen2_05_VLM" and len(inputs) >= 4:
            feed: dict[str, Any] = {
                "inputs_embeds":      inputs[0],
                "mlp1_output":        inputs[1],
                "path_route_queries": inputs[3],
            }
            # Truncate to static shapes where needed
            session_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
            for name, arr in feed.items():
                static = session_shapes.get(name)
                if static is None or not hasattr(arr, "shape"):
                    continue
                slices = tuple(
                    slice(0, d) if isinstance(d, int) and d > 0 and arr.shape[i] > d
                    else slice(None)
                    for i, d in enumerate(static)
                )
                if any(s != slice(None) for s in slices):
                    feed[name] = arr[slices]
            return feed

        # Generic: positional zip
        if len(inputs) == len(input_names):
            return dict(zip(input_names, inputs))

    # Scalar / single-array fallback
    return {input_names[0]: inputs}


def _compute_sqnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Signal-to-Quantization-Noise Ratio in dB."""
    error = np.sum((ref - pred) ** 2)
    if error < 1e-12:
        return float("inf")
    signal = np.sum(ref ** 2)
    return float(10 * np.log10(signal / error))


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

def make_eval_callback(dataloader, module_alias: str, output_names: list[str]):
    """Return a callable (session) → float (mean SQNR over full dataloader).

    Mirrors Quanter.eval_callback_full from quant.py.
    """
    def _eval(session: ort.InferenceSession, num_batches: int | None = None) -> float:
        sqnr_list: list[float] = []
        for batch_idx, batch in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            inp_raw = batch["input"]
            out_raw = batch["output"]

            # Determine batch size
            if isinstance(inp_raw, (list, tuple)):
                bs = int(inp_raw[0].shape[0])
            else:
                bs = int(inp_raw.shape[0])

            for idx in range(bs):
                if isinstance(inp_raw, (list, tuple)):
                    inputs  = [_to_numpy(x[idx]) for x in inp_raw]
                    outputs = [_to_numpy(x[idx]) for x in out_raw]
                else:
                    inputs  = _to_numpy(inp_raw[idx])
                    outputs = [_to_numpy(out_raw[idx])]

                feed     = _make_feed(session, inputs, module_alias)
                ort_outs = session.run(None, feed)

                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                for ort_out, ref_out in zip(ort_outs, outputs):
                    if ref_out is not None:
                        sqnr_list.append(_compute_sqnr(ort_out, ref_out))

        return float(np.mean(sqnr_list)) if sqnr_list else 0.0

    return _eval


# ---------------------------------------------------------------------------
# Enabling loop
# ---------------------------------------------------------------------------

def run_enabling_loop(
    sim: QuantizationSimModel,
    assigned_layers: list[str],
    op_name_to_quantizers: dict[str, list],
    eval_fn,
    logger: logging.Logger,
) -> dict[str, float]:
    """Enable one layer at a time and record the eval (SQNR) score.

    Algorithm
    ---------
    1. Snapshot the enabled / dtype / bitwidth / op_mode of every quantizer.
    2. Disable ALL quantizers.
    3. For each assigned layer:
         a. Re-enable that layer's quantizers (restore snapshotted state).
         b. score = eval_fn(sim.session)
         c. Disable that layer's quantizers again.
    4. Restore all quantizers to their original snapshotted state.

    This is the inner loop of AIMET's
    perform_per_layer_analysis_by_enabling_quantizers, executed only on the
    assigned subset so multiple workers can run in parallel.
    """
    # ── 1. Snapshot ──────────────────────────────────────────────────────────
    # id(q) → (enabled, data_type, bitwidth, op_mode)
    snapshot: dict[int, tuple] = {}
    for q in sim.qc_quantize_op_dict.values():
        try:
            bw = q.bitwidth
        except AttributeError:
            bw = 8   # safe default if attribute name differs across AIMET versions
        snapshot[id(q)] = (q.enabled, q.data_type, bw, q.op_mode)

    # ── 2. Disable all ───────────────────────────────────────────────────────
    for q in sim.qc_quantize_op_dict.values():
        q.enabled = False

    results: dict[str, float] = {}

    # ── 3. Enabling loop ─────────────────────────────────────────────────────
    for layer_name in assigned_layers:
        qs = op_name_to_quantizers.get(layer_name, [])
        if not qs:
            logger.warning("No quantizers found for layer %s — skipping.", layer_name)
            continue

        # Enable this layer's quantizers, restoring their calibrated state
        for q in qs:
            orig = snapshot.get(id(q))
            if orig is None:
                continue
            orig_enabled, orig_dtype, orig_bw, orig_mode = orig
            if not orig_enabled:
                continue   # was disabled by htp_v81 config — do not enable
            q.enabled   = True
            q.data_type = orig_dtype
            q.set_bitwidth(orig_bw)
            q.op_mode   = OpMode.quantizeDequantize

        # Evaluate with only this layer quantized
        try:
            score = eval_fn(sim.session)
        except Exception as exc:
            logger.warning("eval failed for layer %s: %s", layer_name, exc)
            score = float("nan")

        results[layer_name] = score
        logger.debug("  %-52s  SQNR=%8.2f dB", layer_name, score)

        # Disable this layer's quantizers before moving to the next
        for q in qs:
            q.enabled = False

    # ── 4. Restore original state ────────────────────────────────────────────
    for q in sim.qc_quantize_op_dict.values():
        orig = snapshot.get(id(q))
        if orig is None:
            continue
        orig_enabled, orig_dtype, orig_bw, orig_mode = orig
        q.enabled = orig_enabled
        if orig_enabled:
            q.data_type = orig_dtype
            q.set_bitwidth(orig_bw)
            q.op_mode   = orig_mode

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity scan worker")
    parser.add_argument("--onnx-path",      required=True,
                        help="Path to the exported worker_sim.onnx")
    parser.add_argument("--encodings-path", required=True,
                        help="Path to the exported worker_sim.encodings")
    parser.add_argument("--module-alias",   required=True,
                        help="Canonical module alias (e.g. InternViT, Qwen2_05_VLM)")
    parser.add_argument("--data-path",      required=True,
                        help="Intermediate data path for the dataloader")
    parser.add_argument("--qscheme-key",    required=True,
                        help="Quantization scheme key (fp16|w8a16|w8a8|w4a16|w4a4)")
    parser.add_argument("--layers-file",    required=True,
                        help="JSON file containing the list of layer names to scan")
    parser.add_argument("--output-file",    required=True,
                        help="JSON file to write {layer_name: sqnr_score} results to")
    parser.add_argument("--gpu",            type=int, default=0,
                        help="CUDA device id (device 0 of CUDA_VISIBLE_DEVICES env)")
    parser.add_argument("--no-cpu-fallback", action="store_true", default=False,
                        help="Exit with rc=1 on GPU OOM instead of falling back to CPU. "
                             "Prevents workers from running for hours on CPU when VRAM is "
                             "exhausted. The parent process can then detect the failure and "
                             "reduce parallelism.")
    args = parser.parse_args()

    log.info(
        "Worker starting: module=%s  qscheme=%s  gpu=%d",
        args.module_alias, args.qscheme_key, args.gpu,
    )

    # ── Load assigned layer list ─────────────────────────────────────────────
    with open(args.layers_file) as f:
        assigned_layers: list[str] = json.load(f)
    log.info("Assigned layers: %d", len(assigned_layers))

    # ── ORT providers ────────────────────────────────────────────────────────
    providers = _build_providers(args.gpu)
    log.info("ORT providers: %s  (ort_available=%s)", providers, ort.get_available_providers())

    # ── Resolve qscheme types ────────────────────────────────────────────────
    if args.qscheme_key not in QSCHEME_TYPES:
        raise ValueError(
            f"Unknown qscheme_key: {args.qscheme_key!r}. "
            f"Valid keys: {list(QSCHEME_TYPES)}"
        )
    param_type, activation_type = QSCHEME_TYPES[args.qscheme_key]

    # ── Load ONNX + create QuantSim ──────────────────────────────────────────
    log.info("Loading ONNX: %s", args.onnx_path)
    onnx_model = onnx.load(args.onnx_path)

    try:
        sim = QuantizationSimModel(
            model=onnx_model,
            quant_scheme=QuantScheme.post_training_tf,
            config_file="htp_v81",
            param_type=param_type,
            activation_type=activation_type,
            providers=providers,
        )
    except OrtRuntimeException as exc:
        # Any ORT GPU init failure (CUBLAS_STATUS_ALLOC_FAILED, BFCArena OOM, etc.)
        if getattr(args, "no_cpu_fallback", False):
            log.error(
                "GPU session init failed (%s). "
                "--no-cpu-fallback is set — exiting with rc=1 so the parent "
                "process can detect the OOM and reduce parallelism. "
                "Increase --cushion-gb or --sn-per-worker to prevent this.",
                str(exc)[:300],
            )
            sys.exit(1)
        # Fall back to CPU so this worker still completes its layer subset.
        # WARNING: CPU inference for large layer counts is extremely slow
        # (hours instead of minutes). Use --no-cpu-fallback to fail fast
        # and let the parent reduce parallelism instead.
        log.warning(
            "GPU session init failed (%s). "
            "Falling back to CPUExecutionProvider for this worker. "
            "This will be very slow — consider re-running with a larger "
            "--sn-per-worker or --cushion-gb to prevent GPU OOM, or pass "
            "--no-cpu-fallback to fail fast instead.",
            str(exc)[:300],
        )
        providers = ["CPUExecutionProvider"]
        onnx_model = onnx.load(args.onnx_path)   # reload — model may be mutated
        sim = QuantizationSimModel(
            model=onnx_model,
            quant_scheme=QuantScheme.post_training_tf,
            config_file="htp_v81",
            param_type=param_type,
            activation_type=activation_type,
            providers=providers,
        )

    # ── Load calibrated encodings ────────────────────────────────────────────
    # Try the path as-is first; fall back to appending ".json" if needed.
    encodings_path = args.encodings_path
    if not Path(encodings_path).exists():
        alt = encodings_path + ".json"
        if Path(alt).exists():
            encodings_path = alt
        else:
            raise FileNotFoundError(
                f"Encodings file not found: {encodings_path} (also tried {alt})"
            )
    log.info("Loading encodings: %s", encodings_path)
    # strict=False: allows bitwidth mismatches between the fresh QuantSim (base
    # qscheme, e.g. all int8) and the exported encodings, which may reflect a
    # mixed-precision state from a previous MMP step (some quantizers at bw=16).
    # AIMET will update each quantizer's bitwidth/scale/offset to match the
    # encodings rather than raising an AssertionError.
    load_encodings_to_sim(sim, encodings_path, strict=False)

    # ── Build dataloader ─────────────────────────────────────────────────────
    log.info("Building dataloader for module alias: %s", args.module_alias)
    full_dataloader = create_module_dataloader(
        intermediate_data_path=args.data_path,
        module_name=args.module_alias,
        batch_size=1,
        shuffle=False,
    )
    out_names = _output_names(args.module_alias)
    eval_fn   = make_eval_callback(full_dataloader, args.module_alias, out_names)

    # ── Build op_name_to_quantizers for assigned layers only ─────────────────
    assigned_set = set(assigned_layers)
    op_name_to_quantizers: dict[str, list] = {}
    for op in sim.connected_graph.ordered_ops:
        if op.name_op not in assigned_set:
            continue
        in_qs, out_qs, param_qs = sim.get_op_quantizers(op)
        all_qs = list(in_qs) + list(out_qs) + list(param_qs.values())
        if all_qs:
            op_name_to_quantizers[op.name_op] = all_qs

    log.info(
        "Quantizable assigned layers: %d / %d",
        len(op_name_to_quantizers), len(assigned_layers),
    )

    # ── Run enabling loop ────────────────────────────────────────────────────
    results = run_enabling_loop(
        sim=sim,
        assigned_layers=assigned_layers,
        op_name_to_quantizers=op_name_to_quantizers,
        eval_fn=eval_fn,
        logger=log,
    )

    # ── Write results ────────────────────────────────────────────────────────
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results written to %s  (%d layers scored)", args.output_file, len(results))


if __name__ == "__main__":
    main()
