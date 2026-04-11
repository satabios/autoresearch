import os
import sys
import logging
from pathlib import Path
from typing import Any

import numpy as np
from prettytable import PrettyTable
import torch
import onnx
import onnxruntime as ort
from tqdm import tqdm
from onnxruntime import InferenceSession
import aimet_onnx
from aimet_onnx import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme
from pipeline.data_loader import create_module_dataloader
from pipeline.metrics import _compute_sqnr

# Add project root to sys.path before local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["HF_HOME"] = "./"


# ---------------------------------------------------------------------------
# ORT execution providers
# ---------------------------------------------------------------------------
def _build_ort_providers(gpu_id: int = 0) -> list:
    if torch.cuda.is_available():
        return [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

ORT_PROVIDERS: list = _build_ort_providers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.detach().cpu().numpy()
    return tensor


# ---------------------------------------------------------------------------
# Calibration constants & module aliases (mirrors quant.py)
# ---------------------------------------------------------------------------
CALIB_SAMPLES = 50

MODULE_ALIAS: dict[str, str] = {
    "InternViT300M_Pixel_Unshuffle_MLP1":                           "InternViT",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA":               "Qwen2_05_VLM",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_katana": "Qwen2_05_VLM",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_sabre":  "Qwen2_05_VLM",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_oob":    "Qwen2_05_VLM",
}

# None = generic single-input passthrough (fused ONNX has one input named "input").
MODULE_ORT_INPUT_ALIAS: dict[str, str | None] = {
    "InternViT": None,
}


def _build_session_inputs(session: InferenceSession, module_alias: str, inputs: Any) -> dict:
    """Build ORT input feed using quant.py's canonical input mapping.

    Used for both calibration and full-dataset evaluation so both paths use
    the same input layout as the production pipeline.

    Also truncates any input whose runtime shape exceeds the session's static
    shape along each axis — mirrors quant.py::Quanter._make_feed truncation
    logic that handles variable-length sequences (e.g. inputs_embeds with 547
    tokens when the ONNX was exported with a fixed shape of 546).
    """
    ort_key = MODULE_ORT_INPUT_ALIAS.get(module_alias, module_alias)
    if ort_key is None:
        return {session.get_inputs()[0].name: inputs}
    input_map = {
        "non_CoT_token_interleaver": lambda d: {
            "input_embeds":       d[0],
            "vit_embeds":         d[1],
            "input_ids":          d[2],
            "path_route_queries": d[3],
        },
        "InternViT": lambda d: {
            "images":        d[0],
            "inputs_embeds": d[1],
        },
        "Qwen2_05_VLM": lambda d: {
            "inputs_embeds":      d[0],
            "mlp1_output":        d[1],
            "path_route_queries": d[3],  # d[2] is input_ids — not an ONNX input
        },
    }
    builder = input_map.get(ort_key)
    feed = builder(inputs) if builder is not None else {session.get_inputs()[0].name: inputs}

    # Truncate inputs that exceed the session's static shape (e.g. seq_len 547 → 546).
    session_input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
    for name, arr in feed.items():
        static_shape = session_input_shapes.get(name)
        if static_shape is None or not hasattr(arr, "shape"):
            continue
        slices = tuple(
            slice(0, dim) if (isinstance(dim, int) and dim > 0 and arr.shape[i] > dim)
            else slice(None)
            for i, dim in enumerate(static_shape)
        )
        if any(s != slice(None) for s in slices):
            feed[name] = arr[slices]
    return feed


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

table = PrettyTable()
table.field_names = ["Output Name", "SQNR (dB)", "MSE", "Cosine Sim", "Shape"]
table.align["Output Name"] = "l"
table.align["SQNR (dB)"] = "r"
table.align["MSE"] = "r"
table.align["Cosine Sim"] = "r"
table.align["Shape"] = "l"


def _compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute global cosine similarity (flattened). Matches quant.py."""
    a_f = a.flatten().to(torch.float64)
    b_f = b.flatten().to(torch.float64)
    norm_a = torch.linalg.norm(a_f)
    norm_b = torch.linalg.norm(b_f)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(torch.dot(a_f, b_f) / (norm_a * norm_b))


def eval_sim_full_dataset(
    session: InferenceSession,
    full_dataloader: torch.utils.data.DataLoader,
    module_alias: str,
    output_names: list[str],
    label: str,
) -> None:
    """Evaluate *session* over the full dataset and add mean metrics to the results table.

    Mirrors quant.py::Quanter.eval_sim — iterates the full dataloader, accumulates
    per-sample SQNR/MSE/CosSim for each output, then adds one row per output to
    the global PrettyTable with dataset-level means.
    """
    per_output_sqnr:  dict[str, list[float]] = {name: [] for name in output_names}
    per_output_mse:   dict[str, list[float]] = {name: [] for name in output_names}
    per_output_cos:   dict[str, list[float]] = {name: [] for name in output_names}
    per_output_shape: dict[str, tuple]       = {}

    for batch in tqdm(full_dataloader, desc=label, leave=False):
        inp = batch["input"]
        out = batch["output"]

        # Extract sample 0 from the collated batch (DataLoader batch_size=1).
        # Multi-input modules (e.g. Qwen2_05_VLM): inp is a list of tensors.
        # Single-input modules (e.g. InternViT):    inp is a single tensor.
        if isinstance(inp, (list, tuple)):
            inputs  = [to_numpy(x[0]) for x in inp]
            outputs = [x[0].to(torch.float32).to(torch.device("cuda")) for x in out]
        else:
            inputs  = to_numpy(inp[0])
            outputs = [out[0].to(torch.float32).to(torch.device("cuda"))]

        feed     = _build_session_inputs(session, module_alias, inputs)
        ort_outs = session.run(None, feed)

        for name, ort_out, ref_out in zip(output_names, ort_outs, outputs):
            pred = torch.from_numpy(ort_out).to(torch.device("cuda"))
            per_output_sqnr[name].append(float(_compute_sqnr(pred, ref_out)))
            per_output_mse[name].append(float(torch.mean((pred - ref_out) ** 2)))
            per_output_cos[name].append(_compute_cosine_similarity(pred, ref_out))
            per_output_shape[name] = tuple(ort_out.shape)

    for name in output_names:
        if per_output_sqnr[name]:
            table.add_row([
                f"{label}:{name}",
                f"{float(np.mean(per_output_sqnr[name])):.2f}",
                f"{float(np.mean(per_output_mse[name])):.6e}",
                f"{float(np.mean(per_output_cos[name])):.6f}",
                str(per_output_shape.get(name, "?")),
            ])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

data_path = (
    "/local/mnt2/workspace2/users/sathya/dataset/SimLingo/Closed-Loop/Bench2Drive220/"
    "Intermediate_Results/without-CoT/simlingo_b2d_traj-FP32/debug_viz/simlingo/"
    "iter_013.ckpt/leaderboard/bench2drive220_0_simlingo_traj_2026_02_10_15_40_34/"
    "intermediate_data/"
)

modules = [
    # "vit",
    # "InternViT",
    # "InternViT_Interleaving",
    # "pixel_unshuffle",
    # "non_CoT_token_interleaver",
    # "llm_ar32",
    "InternViT300M_Pixel_Unshuffle_MLP1",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA",
    # "mlp1",
]

# InternViT300M_Pixel_Unshuffle_MLP1 : fp16, w8a16, w8a8
# nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA : fp16, w8a16, w4a16

# Per-module QuantSim variants
MODULE_QVARIANTS: dict[str, list[str]] = {
    "InternViT300M_Pixel_Unshuffle_MLP1":             ["fp16", "w8a16", "w8a8"],
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA": ["fp16", "w8a16", "w4a16"],
}
qtypes = {
    "fp16":  (aimet_onnx.float16, aimet_onnx.float16),
    "w8a16": (aimet_onnx.int8,    aimet_onnx.int16),
    "w8a8":  (aimet_onnx.int8,    aimet_onnx.int8),
    "w4a16": (aimet_onnx.int4,    aimet_onnx.int16),
}

ORT      = False   # set True to also run plain FP32 ORT baseline
QuantSim = True

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

for module_name in modules:
    print("Processing module:", module_name)

    # Resolve canonical alias used by the dataloader (mirrors quant.py MODULE_ALIAS)
    module_alias = MODULE_ALIAS.get(module_name, module_name)

    # Determine output names for this module (mirrors quant.py)
    output_names = (
        ["route", "speed"]
        if module_alias in ("llm_ar32", "Qwen2_05_VLM")
        else ["output"]
    )

    # Build full dataloader (entire dataset) and calibration subset (50 samples).
    # Mirrors quant.py::Quanter.build_dataloaders.
    full_dataloader = create_module_dataloader(
        intermediate_data_path=data_path,
        module_name=module_alias,
        batch_size=1,
        shuffle=False,
    )
    _calib_subset = torch.utils.data.Subset(
        full_dataloader.dataset,
        indices=range(min(CALIB_SAMPLES, len(full_dataloader.dataset))),
    )
    calib_dataloader = torch.utils.data.DataLoader(
        _calib_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Per-variant evaluation
    # ------------------------------------------------------------------
    for qvariant in MODULE_QVARIANTS.get(module_name, ["fp16", "w8a16", "w8a8"]):
        # NOTE: Select the correct ONNX variant here.
        # Options (uncomment the one you want to evaluate):
        #   V0 (oob):    f"{module_name}_v0.onnx"
        #   V1 (sabre):  f"{module_name}_Piqaro_sabre_v1.onnx"
        #   V2 (katana): f"{module_name}_Piqaro_katana_v2.onnx"
        onnx_path = (
            f"/local/mnt/workspace/users/sathya/projects/simlingo/"
            f"quantization/onnx/fused/{module_name}_v0.onnx"#_v0.onnx"#_Piqaro_katana_v2.onnx"
        )

        label = f"{module_name} ({qvariant})"

        if qvariant == "":
            # ----------------------------------------------------------
            # FP32 ORT baseline — evaluate over full dataset
            # ----------------------------------------------------------
            if ORT:
                fp32_session = ort.InferenceSession(onnx_path, providers=ORT_PROVIDERS)
                eval_sim_full_dataset(
                    fp32_session, full_dataloader, module_alias, output_names,
                    f"{module_name} (fp32)",
                )

        else:
            # ----------------------------------------------------------
            # QuantSim path — AIMET post-training quantization
            # ----------------------------------------------------------
            if QuantSim:
                param_type, activation_type = qtypes[qvariant]

                sim = QuantizationSimModel(
                    model=onnx.load(onnx_path),
                    quant_scheme=QuantScheme.post_training_tf,
                    config_file="htp_v81",
                    param_type=param_type,
                    activation_type=activation_type,
                    providers=ORT_PROVIDERS,
                )

                # Calibrate: compute encodings over the 50-sample calib subset.
                # Mirrors quant.py's forward_pass_callback.
                def forward_pass_callback(session, args=None):
                    for batch in calib_dataloader:
                        inp = batch["input"]
                        if isinstance(inp, (list, tuple)):
                            inputs = [to_numpy(x[0]) for x in inp]
                        else:
                            inputs = to_numpy(inp[0])
                        feed = _build_session_inputs(session, module_alias, inputs)
                        session.run(None, feed)

                sim.compute_encodings(forward_pass_callback)

                # Evaluate over the full dataset — mirrors quant.py::Quanter.eval_sim.
                # Use try/finally so del sim always runs even if an exception is raised,
                # ensuring AIMET's OrtInferenceSession.__del__ fires while the logging
                # module is still alive (suppresses shutdown-time AttributeError noise).
                try:
                    eval_sim_full_dataset(
                        sim.session, full_dataloader, module_alias, output_names, label
                    )
                finally:
                    del sim

print(table)
