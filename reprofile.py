"""reprofile.py — Post-run quantization profile editor.

Reuses existing quantization artifacts (encodings, sensitivity JSONs) to create
a new mixed-precision profile by flipping layers based on either:
  A) A user-specified sensitivity SQNR threshold  (--threshold), or
  B) An explicit list of layer names              (--layers-file or --layers).

Then re-computes encodings, exports the new profile, and evaluates it against
the full dataset.  A comparison table is printed showing the original run's
fp16 / base / MMP metrics alongside the reprofiled result.

Usage examples
--------------

  # Threshold mode — flip layers with SQNR < 70 dB to fp16
  python reprofile.py \\
      --module InternViT300M_Pixel_Unshuffle_MLP1 --optimized 1 \\
      --qscheme w8a16 --target-dtype fp16 --threshold 70.0

  # Threshold mode — use step-2 sensitivity data explicitly
  python reprofile.py \\
      --module InternViT300M_Pixel_Unshuffle_MLP1 --optimized 1 \\
      --qscheme w8a8 --target-dtype fp16 --threshold 71.5 --sensitivity-step 2

  # Layer-list mode — flip layers listed in a text file
  python reprofile.py \\
      --module InternViT300M_Pixel_Unshuffle_MLP1 --optimized 1 \\
      --qscheme w8a16 --target-dtype fp16 --layers-file layers.txt

  # Layer-list mode — pass layer names inline (no file needed)
  python reprofile.py \\
      --module InternViT300M_Pixel_Unshuffle_MLP1 --optimized 1 \\
      --qscheme w8a16 --target-dtype fp16 \\
      --layers "/vit/encoder/layers.0/attn/Softmax_xx_clone_26" \\
               "/vit/encoder/layers.1/attn/qkv/MatMul__mm_default_to_conv2d_xx_clone_6_2"

  # Start from base quantized encodings instead of MMP
  python reprofile.py ... --from-base

  # Skip the comparison table (faster if final_eval_results.json is absent)
  python reprofile.py ... --no-compare

Flags summary
-------------
  --threshold FLOAT        Mode A: flip layers whose sensitivity SQNR < FLOAT
  --sensitivity-step N     Which per_layer_sensitivity_stepN.json to use
                           (auto-picks lowest available if omitted)
  --sensitivity-json PATH  Override sensitivity JSON path directly
  --layers-file PATH       Mode B: text file, one layer name per line
  --layers NAME [NAME …]   Mode B: inline layer names (space-separated)
  --from-base              Load base quantized encodings instead of MMP
  --no-compare             Skip comparison table against final_eval_results.json
  --output-dir PATH        Custom output directory (auto-timestamped if omitted)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# Add project root and quantization dir to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

import aimet_onnx
from aimet_onnx import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme, QuantizationDataType
from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.qc_quantize_op import OpMode

# quant.py runs _parse_args() at module level via parse_known_args().
# Temporarily strip sys.argv down to just the script name so that
# quant.py's parser doesn't intercept --help or any reprofile.py flags.
_saved_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from quant import (
    Quanter,
    MODULE_ALIAS,
    MODULE_ORT_INPUT_ALIAS,
    MODULE_ONNX_SUFFIX,
    setup_logger,
    _get_module_file_handler,
    print_mmp_layer_table,
)
sys.argv = _saved_argv

os.environ["HF_HOME"] = "./"

# ---------------------------------------------------------------------------
# Constants (mirror quant.py / sensitivity_worker.py to avoid import issues)
# ---------------------------------------------------------------------------

QSCHEMES = {
    "fp16":  (aimet_onnx.float16, aimet_onnx.float16),
    "w8a16": (aimet_onnx.int8,    aimet_onnx.int16),
    "w8a8":  (aimet_onnx.int8,    aimet_onnx.int8),
    "w4a16": (aimet_onnx.int4,    aimet_onnx.int16),
    "w4a4":  (aimet_onnx.int4,    aimet_onnx.int4),
}

# target-dtype string → quantizer settings
TARGET_DTYPE_MAP = {
    "fp16":  (QuantizationDataType.float, 16),
    "w8a16": {"param": (QuantizationDataType.int, 8), "act": (QuantizationDataType.int, 16)},
    "w8a8":  (QuantizationDataType.int, 8),
    "w4a16": {"param": (QuantizationDataType.int, 4), "act": (QuantizationDataType.int, 16)},
    "int16": (QuantizationDataType.int, 16),
    "int8":  (QuantizationDataType.int, 8),
    "int4":  (QuantizationDataType.int, 4),
}

OPT_TAGS    = {0: "oob",    1: "sabre",                     2: "katana"}
ONNX_SUFFIX = {0: "_v0.onnx", 1: "_Piqaro_sabre_v1.onnx",  2: "_Piqaro_katana_v2.onnx"}
ONNX_BASE   = "/local/mnt/workspace/users/sathya/projects/simlingo/quantization/onnx/fused"
RESULTS_BASE = "/local/mnt/workspace/users/sathya/projects/simlingo/quantization/quant_analyzer_results"
DATA_PATH    = "/local/mnt2/workspace2/users/sathya/dataset/SimLingo/Closed-Loop/Bench2Drive220/Intermediate_Results/without-CoT/simlingo_b2d_traj-FP32/debug_viz/simlingo/iter_013.ckpt/leaderboard/bench2drive220_0_simlingo_traj_2026_02_10_15_40_34/intermediate_data/"

# ---------------------------------------------------------------------------
# ORT providers
# ---------------------------------------------------------------------------

def _build_ort_providers(gpu_id: int) -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a new quantization profile by flipping layers "
                    "based on a sensitivity threshold or an explicit layer list."
    )
    p.add_argument("--module", type=str, required=True,
                   help="Module name (must match a completed quant.py run)")
    p.add_argument("--optimized", type=int, default=0, choices=[0, 1, 2],
                   help="Optimization variant: 0=oob, 1=sabre, 2=katana")
    p.add_argument("--qscheme", type=str, required=True, choices=list(QSCHEMES.keys()),
                   help="Base quantization scheme of the existing run")
    p.add_argument("--target-dtype", type=str, required=True,
                   choices=list(TARGET_DTYPE_MAP.keys()),
                   help="Target dtype to flip selected layers to")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device id")

    # Mode A: threshold
    p.add_argument("--threshold", type=float, default=None,
                   help="Sensitivity SQNR threshold — layers with score < threshold "
                        "are flipped to --target-dtype")
    p.add_argument("--sensitivity-json", type=str, default=None,
                   help="Path to per_layer_sensitivity_stepN.json. "
                        "Auto-discovered if omitted (uses --sensitivity-step to pick).")
    p.add_argument("--sensitivity-step", type=int, default=None,
                   help="Which per_layer_sensitivity_stepN.json to use when "
                        "auto-discovering (e.g. 1 or 2). Defaults to the lowest "
                        "available step number.")

    # Mode B: layer list
    p.add_argument("--layers-file", type=str, default=None,
                   help="Text file with one layer name per line. "
                        "Lines starting with # and blank lines are ignored.")
    p.add_argument("--layers", type=str, nargs="+", default=None,
                   help="Inline layer names to flip (space-separated). "
                        "Alternative to --layers-file; both may not be used together.")

    # Starting point
    p.add_argument("--from-base", action="store_true", default=False,
                   help="Load base quantized encodings instead of MMP encodings")
    p.add_argument("--no-compare", action="store_true", default=False,
                   help="Skip loading final_eval_results.json for comparison output")

    # Output
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. Default: auto-generated under results base.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(args: argparse.Namespace) -> dict:
    module_name = args.module
    opt_tag = OPT_TAGS[args.optimized]
    qscheme = args.qscheme

    # Original ONNX model
    if module_name in MODULE_ONNX_SUFFIX:
        suffix = MODULE_ONNX_SUFFIX[module_name]
    else:
        suffix = ONNX_SUFFIX[args.optimized]
    onnx_path = f"{ONNX_BASE}/{module_name}{suffix}"

    # Results directory for this module/opt/qscheme
    qscheme_dir = os.path.join(RESULTS_BASE, module_name, opt_tag, qscheme)
    mmp_dir = os.path.join(qscheme_dir, "mmp")

    # Encodings source
    if args.from_base:
        enc_prefix = f"{module_name}_{qscheme}_quantized"
        enc_dir = qscheme_dir
    else:
        enc_prefix = f"{module_name}_{qscheme}_mmp"
        enc_dir = mmp_dir

    # Handle .encodings vs .encodings.json
    encodings_path = os.path.join(enc_dir, f"{enc_prefix}.encodings")
    if not os.path.exists(encodings_path):
        alt = encodings_path + ".json"
        if os.path.exists(alt):
            encodings_path = alt
        else:
            raise FileNotFoundError(
                f"Encodings not found at {encodings_path} or {alt}. "
                f"Run quant.py first, or use --from-base."
            )

    # Exported ONNX from sim.export (QDQ-augmented)
    sim_onnx_path = os.path.join(enc_dir, f"{enc_prefix}.onnx")
    if not os.path.exists(sim_onnx_path):
        raise FileNotFoundError(
            f"Exported sim ONNX not found at {sim_onnx_path}. "
            f"Ensure quant.py exported this model."
        )

    # Sensitivity JSON (auto-discover or user-provided)
    sensitivity_json = args.sensitivity_json
    if sensitivity_json is None and args.threshold is not None:
        # Auto-discover: honour --sensitivity-step if given, else pick lowest available
        desired_step = getattr(args, "sensitivity_step", None)
        if desired_step is not None:
            candidate = os.path.join(qscheme_dir, f"per_layer_sensitivity_step{desired_step}.json")
            if os.path.exists(candidate):
                sensitivity_json = candidate
            else:
                raise FileNotFoundError(
                    f"Requested --sensitivity-step {desired_step} not found: {candidate}. "
                    f"Available steps are in {qscheme_dir}."
                )
        else:
            for step_n in range(1, 5):
                candidate = os.path.join(qscheme_dir, f"per_layer_sensitivity_step{step_n}.json")
                if os.path.exists(candidate):
                    sensitivity_json = candidate
                    break
        if sensitivity_json is None:
            raise FileNotFoundError(
                f"No per_layer_sensitivity_stepN.json found in {qscheme_dir}. "
                f"Provide --sensitivity-json or --sensitivity-step explicitly."
            )

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(qscheme_dir, f"reprofile_{ts}")

    return {
        "onnx_path": onnx_path,
        "sim_onnx_path": sim_onnx_path,
        "encodings_path": encodings_path,
        "sensitivity_json": sensitivity_json,
        "output_dir": output_dir,
        "qscheme_dir": qscheme_dir,
        "opt_tag": opt_tag,
    }


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def select_layers_by_threshold(
    sensitivity_json: str,
    threshold: float,
    valid_op_names: set[str],
    logger: logging.Logger,
) -> list[tuple[str, float]]:
    """Return (layer_name, sqnr_score) for layers with score < threshold."""
    with open(sensitivity_json) as f:
        sensitivity_data = json.load(f)

    selected = []
    for name, score in sensitivity_data.items():
        if name not in valid_op_names:
            continue
        if score < threshold:
            selected.append((name, score))

    selected.sort(key=lambda x: x[1])
    logger.info(
        "Threshold mode: %d / %d layers below threshold %.2f dB",
        len(selected), len(sensitivity_data), threshold,
    )
    return selected


def select_layers_by_name(
    layers_file: str,
    valid_op_names: set[str],
    logger: logging.Logger,
) -> list[str]:
    """Read layer names from a text file. Validate against sim op names."""
    with open(layers_file) as f:
        raw_lines = f.readlines()

    names = []
    warnings = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip trailing commas and quotes (common from copy-pasting JSON arrays)
        line = line.strip('", ')
        if line in valid_op_names:
            names.append(line)
        else:
            warnings.append(line)

    if warnings:
        logger.warning(
            "The following %d layer(s) from %s were NOT found in the sim and will be skipped:\n  %s",
            len(warnings), layers_file, "\n  ".join(warnings),
        )

    logger.info(
        "Layer-list mode: %d layers selected from %s (%d skipped)",
        len(names), layers_file, len(warnings),
    )
    return names


# ---------------------------------------------------------------------------
# Quantizer manipulation
# ---------------------------------------------------------------------------

def apply_dtype_to_layer(
    layer_name: str,
    target_step,
    op_name_to_quantizers: dict,
    op_name_to_param_quantizers: dict,
    op_name_to_act_quantizers: dict,
    pre_disabled_ids: set,
) -> int:
    """Apply *target_step* to all active quantizers of *layer_name*.

    target_step can be:
      (QuantizationDataType, bw)                   — promote ALL quantizers
      {"param": (dtype, bw), "act": (dtype, bw)}   — split param/act
    """
    changed = 0
    if isinstance(target_step, dict):
        param_step = target_step.get("param")
        act_step = target_step.get("act")
        for q in op_name_to_param_quantizers.get(layer_name, []):
            if id(q) in pre_disabled_ids:
                continue
            dtype, bw = param_step
            q.enabled = True
            q.data_type = dtype
            q.set_bitwidth(bw)
            q.op_mode = OpMode.quantizeDequantize
            changed += 1
        for q in op_name_to_act_quantizers.get(layer_name, []):
            if id(q) in pre_disabled_ids:
                continue
            dtype, bw = act_step
            q.enabled = True
            q.data_type = dtype
            q.set_bitwidth(bw)
            q.op_mode = OpMode.quantizeDequantize
            changed += 1
    else:
        dtype, bw = target_step
        for q in op_name_to_quantizers.get(layer_name, []):
            if id(q) in pre_disabled_ids:
                continue
            q.enabled = True
            q.data_type = dtype
            q.set_bitwidth(bw)
            q.op_mode = OpMode.quantizeDequantize
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Comparison table helpers
# ---------------------------------------------------------------------------

def _fmt_metric(m: dict | None) -> str:
    """Format a metrics dict as 'SQNR / MSE / CosSim'."""
    if m is None:
        return "     —     /     —     /     —    "
    cos = m.get("cosine_sim")
    cos_str = f"{cos:.4f}" if cos is not None else "  —   "
    return f"{m['sqnr_db']:7.2f} dB / {m['mse']:.3e} / {cos_str}"


def print_comparison_table(
    module_name: str,
    qscheme: str,
    opt_tag: str,
    original_results: dict | None,
    reprofile_metrics: dict,
    n_flipped: int,
    target_dtype: str,
    from_base: bool,
    logger: logging.Logger,
) -> None:
    """Print a side-by-side comparison of original run vs reprofiled model.

    *original_results* is the dict loaded from final_eval_results.json.
    Rows shown (when available):
      fp32 ℹ  — informational baseline (not on device)
      fp16 ◀  — on-device reference
      <qscheme> base  — base quantized (before MMP)
      <qscheme> MMP   — after MMP (the starting point for reprofile)
      reprofile       — this run's result
    """
    col_w = 38
    sep = "─" * (18 + col_w + 2)

    def _row(label: str, m: dict | None, note: str = "") -> str:
        metric_str = _fmt_metric(m)
        note_part = f"  ← {note}" if note else ""
        return f"│ {label:<16} │ {metric_str:<{col_w}} │{note_part}"

    lines = [
        f"┌{sep}┐",
        f"│ Reprofile comparison: {module_name} [{opt_tag}] / {qscheme:<{len(sep)-26}}│",
        f"├{'─'*18}┬{'─'*col_w}┤",
        f"│ {'Variant':<16} │ {'SQNR (dB) / MSE / CosSim':<{col_w}} │",
        f"├{'─'*18}┼{'─'*col_w}┤",
    ]

    if original_results:
        # FP32 informational row
        fp32_m = original_results.get("fp32_info")
        if fp32_m:
            lines.append(_row("fp32 ℹ", fp32_m, "not on device"))

        # FP16 reference row
        fp16_data = original_results.get("fp16", {})
        fp16_m = fp16_data.get("quant") if isinstance(fp16_data, dict) else None
        if fp16_m:
            lines.append(_row("fp16 ◀ ref", fp16_m, "on-device reference"))

        lines.append(f"├{'─'*18}┼{'─'*col_w}┤")

        # Base quantized row
        qs_data = original_results.get(qscheme, {})
        if isinstance(qs_data, dict):
            base_m = qs_data.get("quant")
            mmp_m  = qs_data.get("mmp")
            if base_m:
                lines.append(_row(f"{qscheme} base", base_m))
            if mmp_m:
                lines.append(_row(f"{qscheme} MMP", mmp_m, "starting point" if not from_base else ""))

        lines.append(f"├{'─'*18}┼{'─'*col_w}┤")

    # Reprofile row
    lines.append(_row(
        "reprofile",
        reprofile_metrics,
        f"{n_flipped} layers → {target_dtype}",
    ))
    lines.append(f"└{'─'*18}┴{'─'*col_w}┘")

    logger.info("Comparison table:\n%s", "\n".join(lines))


def main():
    args = parse_args()

    # Validate mode — exactly one of: --threshold, --layers-file, --layers
    has_threshold  = args.threshold is not None
    has_layers_file = args.layers_file is not None
    has_layers_inline = args.layers is not None and len(args.layers) > 0

    n_modes = sum([has_threshold, has_layers_file, has_layers_inline])
    if n_modes != 1:
        print(
            "ERROR: Specify exactly one of --threshold, --layers-file, or --layers.",
            file=sys.stderr,
        )
        sys.exit(1)

    has_layers = has_layers_file or has_layers_inline

    # Setup
    gpu_id = args.gpu
    ort_providers = _build_ort_providers(gpu_id)
    opt_tag = OPT_TAGS[args.optimized]
    module_alias = MODULE_ALIAS.get(args.module, args.module)

    log_name = f"reprofile.{args.module}.{opt_tag}.{args.qscheme}"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger(log_name)

    paths = resolve_paths(args)
    os.makedirs(paths["output_dir"], exist_ok=True)

    logger.info("Module:       %s (alias: %s)", args.module, module_alias)
    logger.info("QScheme:      %s", args.qscheme)
    logger.info("Target dtype: %s", args.target_dtype)
    logger.info("Starting from:%s", "base" if args.from_base else "MMP")
    logger.info("Sim ONNX:     %s", paths["sim_onnx_path"])
    logger.info("Encodings:    %s", paths["encodings_path"])
    logger.info("Output:       %s", paths["output_dir"])

    # ------------------------------------------------------------------
    # Step 1: Build dataloaders
    # ------------------------------------------------------------------
    logger.info("Building dataloaders ...")
    full_dataloader, calib_dataloader = Quanter.build_dataloaders(
        data_path=DATA_PATH,
        module_name=module_alias,
    )
    output_names = (
        ["route", "speed"]
        if module_alias in ("llm_ar32", "Qwen2_05_VLM")
        else ["output"]
    )

    # ------------------------------------------------------------------
    # Step 2: Load QuantSim from existing artifacts
    # ------------------------------------------------------------------
    logger.info("Loading QuantSim from exported artifacts ...")
    param_type, activation_type = QSCHEMES[args.qscheme]
    sim_onnx_model = onnx.load(paths["sim_onnx_path"])

    sim = QuantizationSimModel(
        model=sim_onnx_model,
        quant_scheme=QuantScheme.post_training_tf,
        config_file="htp_v81",
        param_type=param_type,
        activation_type=activation_type,
        providers=ort_providers,
    )
    load_encodings_to_sim(sim, paths["encodings_path"], strict=False)
    logger.info("QuantSim loaded successfully.")

    # ------------------------------------------------------------------
    # Step 3: Build quantizer maps
    # ------------------------------------------------------------------
    op_name_to_quantizers: dict[str, list] = {}
    op_name_to_param_quantizers: dict[str, list] = {}
    op_name_to_act_quantizers: dict[str, list] = {}
    for op in sim.connected_graph.ordered_ops:
        in_qs, out_qs, param_qs = sim.get_op_quantizers(op)
        act_qs_list = list(in_qs) + list(out_qs)
        param_qs_list = list(param_qs.values())
        all_qs = act_qs_list + param_qs_list
        if all_qs:
            op_name_to_quantizers[op.name_op] = all_qs
            op_name_to_param_quantizers[op.name_op] = param_qs_list
            op_name_to_act_quantizers[op.name_op] = act_qs_list

    pre_disabled_ids = {
        id(q) for q in sim.qc_quantize_op_dict.values() if not q.enabled
    }
    valid_op_names = set(op_name_to_quantizers.keys())

    # Classify htp-disabled vs active ops
    htp_disabled_layers = [
        name for name, qs in op_name_to_quantizers.items()
        if all(id(q) in pre_disabled_ids for q in qs)
    ]
    active_ops = [
        name for name in op_name_to_quantizers
        if name not in htp_disabled_layers
    ]

    logger.info(
        "Sim has %d ops with quantizers (%d active, %d htp-disabled)",
        len(op_name_to_quantizers), len(active_ops), len(htp_disabled_layers),
    )

    # ------------------------------------------------------------------
    # Step 4: Select layers to flip
    # ------------------------------------------------------------------
    target_step = TARGET_DTYPE_MAP[args.target_dtype]
    sensitivity_data = None

    if has_threshold:
        logger.info("Mode: threshold (%.2f dB)", args.threshold)
        sensitivity_data_path = paths["sensitivity_json"]
        with open(sensitivity_data_path) as f:
            sensitivity_data = json.load(f)
        logger.info("Loaded sensitivity data from %s", sensitivity_data_path)

        selected = select_layers_by_threshold(
            sensitivity_data_path, args.threshold, set(active_ops), logger,
        )
        layers_to_flip = [name for name, _ in selected]
    elif has_layers_file:
        logger.info("Mode: layer-list (file: %s)", args.layers_file)
        layers_to_flip = select_layers_by_name(
            args.layers_file, valid_op_names, logger,
        )
    else:
        # Inline --layers
        logger.info("Mode: layer-list (inline, %d names)", len(args.layers))
        layers_to_flip = []
        skipped = []
        for raw in args.layers:
            name = raw.strip().strip('", ')
            if name in valid_op_names:
                layers_to_flip.append(name)
            else:
                skipped.append(name)
        if skipped:
            logger.warning(
                "The following %d inline layer(s) were NOT found in the sim and will be skipped:\n  %s",
                len(skipped), "\n  ".join(skipped),
            )
        logger.info(
            "Inline layer-list: %d layers selected (%d skipped)",
            len(layers_to_flip), len(skipped),
        )

    if not layers_to_flip:
        logger.warning("No layers selected for flipping. Nothing to do.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 5: Apply flips
    # ------------------------------------------------------------------
    logger.info("Flipping %d layers to %s ...", len(layers_to_flip), args.target_dtype)
    total_quantizers_changed = 0
    for layer_name in layers_to_flip:
        n = apply_dtype_to_layer(
            layer_name, target_step,
            op_name_to_quantizers,
            op_name_to_param_quantizers,
            op_name_to_act_quantizers,
            pre_disabled_ids,
        )
        score_str = ""
        if sensitivity_data and layer_name in sensitivity_data:
            score_str = f"  SQNR={sensitivity_data[layer_name]:.2f} dB"
        logger.debug("  flip %-60s  → %s%s  (%d quantizers)", layer_name, args.target_dtype, score_str, n)
        total_quantizers_changed += n

    # Ensure all enabled quantizers are in QDQ mode
    for q in sim.qc_quantize_op_dict.values():
        if q.enabled:
            q.op_mode = OpMode.quantizeDequantize

    logger.info(
        "Flipped %d layers (%d quantizers) to %s.",
        len(layers_to_flip), total_quantizers_changed, args.target_dtype,
    )

    # ------------------------------------------------------------------
    # Step 6: Build Quanter for callbacks & evaluation
    # ------------------------------------------------------------------
    quanter = Quanter(
        paths["onnx_path"],
        module_alias,
        static_data=None,
        calib_dataloader=calib_dataloader,
        full_dataloader=full_dataloader,
        output_names=output_names,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Step 7: Re-compute encodings
    # ------------------------------------------------------------------
    logger.info("Re-computing encodings after flips ...")
    sim.compute_encodings(quanter.forward_pass_callback)
    logger.info("Encodings recomputed.")

    # ------------------------------------------------------------------
    # Step 8: Export
    # ------------------------------------------------------------------
    export_prefix = f"{args.module}_{args.qscheme}_reprofile"
    sim.export(path=paths["output_dir"], filename_prefix=export_prefix, export_model=True)
    logger.info(
        "Exported: %s/%s.onnx + .encodings",
        paths["output_dir"], export_prefix,
    )

    # ------------------------------------------------------------------
    # Step 9: Evaluate
    # ------------------------------------------------------------------
    logger.info("Evaluating reprofiled model ...")
    metrics = quanter.eval_sim(sim, label="reprofile")
    logger.info(
        "[reprofile] SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
        metrics["sqnr_db"], metrics["mse"], metrics["cosine_sim"],
    )

    # ------------------------------------------------------------------
    # Step 10: Print per-layer breakdown table
    # ------------------------------------------------------------------
    flipped_layers_info = []
    remaining_layers_info = []
    flipped_set = set(layers_to_flip)

    if sensitivity_data:
        for name in active_ops:
            score = sensitivity_data.get(name, float("nan"))
            if name in flipped_set:
                flipped_layers_info.append((name, score, args.target_dtype))
            else:
                remaining_layers_info.append((name, score))
    else:
        for name in active_ops:
            if name in flipped_set:
                flipped_layers_info.append((name, float("nan"), args.target_dtype))
            else:
                remaining_layers_info.append((name, float("nan")))

    print_mmp_layer_table(
        qscheme_key=args.qscheme,
        sensitivity_data=sensitivity_data or {},
        htp_disabled_layers=htp_disabled_layers,
        flipped_layers=flipped_layers_info,
        remaining_layers=remaining_layers_info,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Step 11: Comparison table against original run metrics
    # ------------------------------------------------------------------
    if not args.no_compare:
        final_results_path = os.path.join(
            RESULTS_BASE, args.module, paths["opt_tag"], "final_eval_results.json"
        )
        original_results = None
        if os.path.exists(final_results_path):
            try:
                with open(final_results_path) as f:
                    original_results = json.load(f)
                logger.debug("Loaded original run results from %s", final_results_path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", final_results_path, exc)
        else:
            logger.debug(
                "No final_eval_results.json found at %s — skipping comparison.",
                final_results_path,
            )

        print_comparison_table(
            module_name=args.module,
            qscheme=args.qscheme,
            opt_tag=paths["opt_tag"],
            original_results=original_results,
            reprofile_metrics=metrics,
            n_flipped=len(layers_to_flip),
            target_dtype=args.target_dtype,
            from_base=args.from_base,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Step 12: Save config and results
    # ------------------------------------------------------------------
    mode_label = (
        "threshold" if has_threshold
        else ("layer_list_file" if has_layers_file else "layer_list_inline")
    )
    config = {
        "module": args.module,
        "module_alias": module_alias,
        "optimized": args.optimized,
        "opt_tag": opt_tag,
        "qscheme": args.qscheme,
        "target_dtype": args.target_dtype,
        "from_base": args.from_base,
        "mode": mode_label,
        "threshold": args.threshold,
        "sensitivity_json": paths.get("sensitivity_json"),
        "sensitivity_step": getattr(args, "sensitivity_step", None),
        "layers_file": args.layers_file,
        "layers_inline": args.layers,
        "layers_flipped": layers_to_flip,
        "n_layers_flipped": len(layers_to_flip),
        "n_quantizers_changed": total_quantizers_changed,
        "encodings_source": paths["encodings_path"],
        "sim_onnx_source": paths["sim_onnx_path"],
    }
    with open(os.path.join(paths["output_dir"], "reprofile_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    results = {
        "sqnr_db": metrics["sqnr_db"],
        "mse": metrics["mse"],
        "cosine_sim": metrics["cosine_sim"],
        "n_layers_flipped": len(layers_to_flip),
        "target_dtype": args.target_dtype,
    }
    with open(os.path.join(paths["output_dir"], "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Done. Results saved to %s", paths["output_dir"])

    del sim


if __name__ == "__main__":
    main()
