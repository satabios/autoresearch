import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnx
import onnxruntime as ort
from tqdm import tqdm
from onnxruntime import InferenceSession
import aimet_onnx
from aimet_onnx import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme, QuantizationDataType
from aimet_onnx.common.utils import CallbackFunc
from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.quant_analyzer import QuantAnalyzer
from aimet_onnx.common.quant_analyzer import export_per_layer_sensitivity_analysis_plot
from pipeline.data_loader import create_module_dataloader
# Add project root to sys.path before local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["HF_HOME"] = "./"

# ---------------------------------------------------------------------------
# CLI — parsed once at module load so both direct runs and subprocess launches
# pick up --module / --optimized / --gpu without any global mutation.
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-module ONNX quantization worker")
    p.add_argument("--module",     type=str,  default=None,
                   help="Module name to quantize (overrides the 'modules' list)")
    p.add_argument("--optimized",  type=int,  default=None,
                   help="Optimization variant: 0=oob, 1=sabre, 2=katana")
    p.add_argument("--gpu",        type=int,  default=None,
                   help="CUDA device id to use (overrides CUDA_VISIBLE_DEVICES)")
    # ---------------------------------------------------------------------------
    # Parallel sensitivity scan flags
    # ---------------------------------------------------------------------------
    p.add_argument("--parallel-sensitivity", action="store_true", default=False,
                   help="Parallelize per-layer sensitivity scan across multiple GPUs")
    p.add_argument("--sensitivity-gpu-ids", type=str, default=None,
                   help="Comma-separated physical GPU IDs for sensitivity workers "
                        "(e.g. '0,1,2,3').  Required when --parallel-sensitivity is set.")
    p.add_argument("--vram-per-gpu", type=float, default=80.0,
                   help="VRAM per GPU in GB used to compute P (default: 80 for A100/H100)")
    p.add_argument("--cushion-gb",   type=float, default=3.0,
                   help="VRAM cushion reserved per GPU in GB (default: 3).  "
                        "Used as static fallback for currgpu_avail when pynvml is unavailable.")
    p.add_argument("--sn-per-worker", type=float, default=0.0,
                   help="Per-worker safety net in GB (spec: sn).  Added to m_u to form "
                        "U = m_u + sn.  Passed by launch_quant.py from "
                        "ARCHITECTURE_SAFETY_NET_GB.  Default 0.0 for backward compat.")
    p.add_argument("--dynamic-scheduling", action="store_true", default=False,
                   help="Use Ray-based dynamic memory scheduler instead of static "
                        "subprocess allocation.  Requires 'ray' to be installed.")
    p.add_argument("--no-live-vram-poll", action="store_true", default=False,
                   help="Disable live pynvml VRAM polling; use static M - cushion instead.")
    p.add_argument("--gpu-headroom-gb", type=float, default=1.0,
                   help="Per-GPU global VRAM headroom in GB reserved before computing "
                        "workers_per_gpu.  Applied as: "
                        "effective_avail = currgpu_avail - gpu_headroom_gb.  "
                        "Covers driver overhead, fragmentation, and processes that "
                        "start after the pynvml poll.  Default: 1.0 GB.")
    # allow_abbrev=False + unknown args ignored so pytest / notebooks don't break
    args, _ = p.parse_known_args()
    return args

_ARGS = _parse_args()

# ---------------------------------------------------------------------------
# ORT execution providers — built from the CLI --gpu flag (or device 0).
# When launched via launch_quant.py the subprocess already has
# CUDA_VISIBLE_DEVICES=<physical_id> set, so device_id=0 always refers to
# the correct physical GPU inside that process.
# ---------------------------------------------------------------------------
def _build_ort_providers(gpu_id: int) -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    print(
        f"[quant] WARNING: CUDAExecutionProvider not in ORT available providers {available}. "
        "Falling back to CPU. Install onnxruntime-gpu and ensure CUDA libs are on LD_LIBRARY_PATH.",
        flush=True,
    )
    return ["CPUExecutionProvider"]

# Resolved once: CLI --gpu wins; fall back to 0.
_GPU_ID: int = _ARGS.gpu if _ARGS.gpu is not None else 0
ORT_PROVIDERS: list = _build_ort_providers(_GPU_ID)

# Startup confirmation — printed before any logger is configured so it always
# appears in the terminal / launch.log regardless of log level.
print(
    f"[quant] ORT providers: {ORT_PROVIDERS}  "
    f"(gpu_id={_GPU_ID}, "
    f"ort_available={ort.get_available_providers()})",
    flush=True,
)




# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_ROOT = Path("/local/mnt/workspace/users/sathya/projects/simlingo/quantization/logs")

# Single shared file handler per module — attached to every logger created for
# that module so all output (summary + per-qscheme + AIMET) lands in one file.
_module_file_handlers: dict[str, logging.FileHandler] = {}


def _get_module_file_handler(module_name: str) -> logging.FileHandler:
    """Return (creating if needed) the single FileHandler for *module_name*.

    All loggers for the same module share this handler so every line —
    summary, per-qscheme, AIMET internals — ends up in one run.log.
    """
    if module_name not in _module_file_handlers:
        log_dir  = LOG_ROOT / module_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        _module_file_handlers[module_name] = fh
    return _module_file_handlers[module_name]


def setup_logger(module_name: str, qscheme_key: str) -> logging.Logger:
    """Create a logger for *module_name* / *qscheme_key*.

    All loggers for the same module share a single FileHandler so every
    line — summary, per-qscheme, AIMET internals — lands in one
    LOG_ROOT/{module_name}/run.log file.

    propagate=False prevents double-printing to the root logger's stderr.
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(f"quant.{module_name}.{qscheme_key}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # do not bubble up to root's stderr handler

    # Shared file handler — same file for every qscheme under this module
    fh = _get_module_file_handler(module_name)
    if fh not in logger.handlers:
        logger.addHandler(fh)

    # Console handler — INFO and above to stdout (added only once)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


class Quanter:
    # Total dataset size is ~367; 50 samples is enough for calibration
    CALIB_SAMPLES = 50

    def __init__(
        self,
        onnx_export_path: str,
        module_name: str,
        static_data: Any,
        calib_dataloader: Any = None,
        full_dataloader: Any = None,
        output_names: list[str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.onnx_export_path  = onnx_export_path
        self.module_name       = module_name
        self.static_data       = static_data
        self.calib_dataloader  = calib_dataloader  # small subset — used by calibration callbacks
        self.full_dataloader   = full_dataloader   # entire dataset — used by per-layer analysis
        self.dataloader        = calib_dataloader  # default for callbacks
        self.logger            = logger or logging.getLogger(f"quant.{module_name}")
        # Default output names per module; can be overridden via constructor
        self.output_names = output_names or (
            ["route", "speed"]
            if module_name in ("llm_ar32", "Qwen2_05_VLM")
            else ["output"]
        )
        self.eval_metrics = {
            "InternViT": {
                "SQNR": 72.47,
                "MSE": 6.582685e-08,
            },
            "Qwen2_05_VLM": [  # Two outputs: route and speed
                {"SQNR": 92.39, "MSE": 3.545714e-08},
                {"SQNR": 77.02, "MSE": 4.603065e-07},
            ],
        }

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def build_dataloaders(
        cls,
        data_path: str,
        module_name: str,
        full_batch_size: int = 1,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create and return (full_dataloader, calib_dataloader).

        full_dataloader  — entire dataset, used for the one-shot MSE loss pass.
        calib_dataloader — shuffled subset of CALIB_SAMPLES, used by the
                           forward_pass and eval callbacks that run repeatedly
                           during QuantAnalyzer analysis.

        Note: torch.utils.data.Subset does not accept a shuffle argument;
        shuffling is applied only on the DataLoader.
        """
        full_dataloader = create_module_dataloader(
            intermediate_data_path=data_path,
            module_name=module_name,
            batch_size=full_batch_size,
            shuffle=False,
        )

        calib_subset = torch.utils.data.Subset(
            full_dataloader.dataset,
            indices=range(min(cls.CALIB_SAMPLES, len(full_dataloader.dataset))),
        )
        calib_dataloader = torch.utils.data.DataLoader(
            calib_subset,
            batch_size=1,
            shuffle=True,   # shuffle here, not on Subset
            num_workers=0,
        )

        return full_dataloader, calib_dataloader

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _to_numpy(self, obj: Any) -> Any:
        """Recursively convert torch tensors to numpy arrays."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, dict):
            return {k: self._to_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_numpy(item) for item in obj]
        return obj

    @staticmethod
    def _compute_sqnr(module_output: np.ndarray, static_output: np.ndarray) -> float:
        """Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB."""
        error = np.sum((static_output - module_output) ** 2)
        if error < 1e-12:
            return float("inf")
        signal_power = np.sum(static_output ** 2)
        return float(10 * np.log10(signal_power / error))

    @staticmethod
    def _compute_mse(module_output: np.ndarray, static_output: np.ndarray) -> float:
        """Compute Mean Squared Error (MSE)."""
        return float(np.mean((module_output - static_output) ** 2))

    @staticmethod
    def _compute_cosine_similarity(module_output: np.ndarray, static_output: np.ndarray) -> float:
        """Compute cosine similarity between flattened output vectors.

        Returns a value in [-1, 1] where 1.0 means identical direction.
        Returns 1.0 when either vector is all-zeros (no meaningful angle).
        """
        a = module_output.flatten().astype(np.float64)
        b = static_output.flatten().astype(np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 1.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _batch_size(self, batch: dict) -> int:
        """Return the number of samples in *batch*.

        ``batch["input"]`` has two shapes depending on the module:
        * Multi-input modules (e.g. Qwen2_05_VLM): a list of N tensors each
          shaped ``[B, ...]``.  The batch size is ``batch["input"][0].shape[0]``.
        * Single-input modules (e.g. mlp1): a single tensor shaped ``[B, ...]``.
          The batch size is ``batch["input"].shape[0]``.
        """
        inp = batch["input"]
        if isinstance(inp, (list, tuple)):
            return int(inp[0].shape[0])
        return int(inp.shape[0])

    def _parse_batch_item(self, batch: dict, idx: int) -> tuple[Any, Any]:
        """Extract and convert sample *idx* from a collated batch to numpy.

        *idx* always indexes the **batch dimension** (axis 0 of each tensor),
        never the number of inputs.  Use ``_batch_size(batch)`` to get the
        correct upper bound for the loop.
        """
        if isinstance(batch["input"], (list, tuple)):
            inputs  = [self._to_numpy(x[idx]) for x in batch["input"]]
            outputs = [self._to_numpy(x[idx]) for x in batch["output"]]
        else:
            inputs  = self._to_numpy(batch["input"][idx])
            outputs = self._to_numpy(batch["output"][idx])
        return inputs, outputs

    def _make_feed(self, session: InferenceSession, inputs: Any) -> dict:
        """Build the ORT input feed dict for *session* from raw *inputs*.

        Two concerns are separated via lookup tables:

        MODULE_ORT_INPUT_ALIAS (keyed on self.module_name / canonical alias)
        -----------------------------------------------------------------------
        * ``None``  → generic single-input passthrough: the raw data is passed
          directly to the session's first (and only) input.  Used for fused
          models like InternViT300M_Pixel_Unshuffle_MLP1 whose ONNX has a
          single input named ``"input"`` and whose dataloader already returns
          a single tensor (not a tuple).
        * a string → use that key to look up the named-input builder in
          ``_build_ort_inputs``.

        Modules not in MODULE_ORT_INPUT_ALIAS fall back to self.module_name,
        which routes through the per-module named-input builders (e.g.
        ``Qwen2_05_VLM`` drops ``input_ids`` at d[2] and maps the rest).
        """
        # Resolve which _build_ort_inputs branch to use.
        # Sentinel None → generic single-input passthrough.
        ort_key = MODULE_ORT_INPUT_ALIAS.get(self.module_name, self.module_name)
        if ort_key is None:
            # The dataloader for this module already returns a single tensor
            # (not a tuple), so pass it straight through to the session's
            # only input.  Do NOT index into it — that would slice the array.
            return {session.get_inputs()[0].name: inputs}
        ort_inputs = self._build_ort_inputs(ort_key, inputs)
        # Generic fallback: _build_ort_inputs returned {"input": data} because
        # ort_key had no special mapping — replace placeholder with actual name.
        if "input" in ort_inputs:
            actual_name = session.get_inputs()[0].name
            return {actual_name: ort_inputs["input"]}
        # Truncate any input whose runtime shape exceeds the session's static
        # shape along each axis.  This handles modules like Qwen2_05_VLM whose
        # inputs_embeds has a variable sequence length (e.g. 546 or 547 tokens)
        # but the ONNX was exported with a fixed static shape.  We truncate
        # rather than pad so no dummy tokens are fed to the model.
        session_input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
        for name, arr in ort_inputs.items():
            static_shape = session_input_shapes.get(name)
            if static_shape is None or not hasattr(arr, "shape"):
                continue
            slices = tuple(
                slice(0, dim) if (isinstance(dim, int) and dim > 0 and arr.shape[i] > dim)
                else slice(None)
                for i, dim in enumerate(static_shape)
            )
            if any(s != slice(None) for s in slices):
                ort_inputs[name] = arr[slices]
        return ort_inputs

    # ---------------------------------------------------------------------------
    # ONNX session
    # ---------------------------------------------------------------------------

    @staticmethod
    def _build_ort_session(onnx_export_path: str) -> ort.InferenceSession:
        """Create and return an ONNX Runtime inference session on the correct device."""
        return ort.InferenceSession(onnx_export_path, providers=ORT_PROVIDERS)

    def _build_ort_inputs(self, module_name: str, static_data: Any) -> dict:
        """Map static data to the correct ONNX input dict for a given module.

        Uses lazy lambdas so index accesses are only evaluated for the matching
        module — prevents IndexError when static_data has fewer elements than
        another module's mapping expects.
        """
        input_map = {
            "non_CoT_token_interleaver": lambda d: {
                "input_embeds":       d[0],
                "vit_embeds":         d[1],
                "input_ids":          d[2],
                "path_route_queries": d[3],
            },
            "InternViT": lambda d: {
                "images":             d[0],
                "inputs_embeds":      d[1],
            },
            "Qwen2_05_VLM": lambda d: {
                "inputs_embeds":      d[0],  # (seq_len, 896) — truncated in _make_feed
                "mlp1_output":        d[1],  # (2, 256, 896)
                "path_route_queries": d[3],  # (1, 30, 896)
            },
        }
        builder = input_map.get(module_name)
        if builder is not None:
            return builder(static_data)
        return {"input": static_data}

    @staticmethod
    def _print_input_shapes(ort_inputs: dict, logger: logging.Logger) -> None:
        """Log the shape of each ONNX input tensor."""
        if "input" in ort_inputs:
            logger.debug("Input shape: %s", ort_inputs["input"].shape)
        else:
            logger.debug("Input shapes:")
            for key, value in ort_inputs.items():
                logger.debug("  %s: %s", key, value.shape)

    # ---------------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------------

    def _log_output_metrics(
        self,
        name: str,
        module_name: str,
        ort_out: np.ndarray,
        sqnr: float,
        mse: float,
        cosine_sim: float,
    ) -> None:
        """Log per-output verification metrics."""
        self.logger.info(
            "Module: %s | Output: %s | Shape: %s | SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
            module_name, name, tuple(ort_out.shape), sqnr, mse, cosine_sim,
        )

    def _evaluate_outputs(
        self,
        ort_outs: list,
        output_data: Any,
        module_name: str,
        output_names: list[str],
        verbose: bool,
    ) -> tuple[float, float, float, dict]:
        """Compute and log SQNR, MSE, and cosine similarity for each output.

        For multi-output modules (e.g. route + speed), metrics are logged
        individually per output at INFO (verbose) or DEBUG (non-verbose) level.
        Returns (mean_sqnr, mean_mse, mean_cos, per_output) where per_output
        maps each output name to its (sqnr, mse, cosine_sim) triple.
        """
        if not isinstance(output_data, (tuple, list)):
            output_data = (output_data,)

        per_sqnr, per_mse, per_cos = [], [], []
        per_output: dict[str, tuple[float, float, float]] = {}
        for name, ort_out, out_data in zip(output_names, ort_outs, output_data):
            if out_data is None:
                continue
            sqnr       = self._compute_sqnr(ort_out, out_data)
            mse        = self._compute_mse(ort_out, out_data)
            cosine_sim = self._compute_cosine_similarity(ort_out, out_data)
            per_sqnr.append(sqnr)
            per_mse.append(mse)
            per_cos.append(cosine_sim)
            per_output[name] = (sqnr, mse, cosine_sim)
            if verbose:
                self._log_output_metrics(name, module_name, ort_out, sqnr, mse, cosine_sim)

        if not per_sqnr:
            return None, None, None, {}
        return float(np.mean(per_sqnr)), float(np.mean(per_mse)), float(np.mean(per_cos)), per_output

    def verify_ort_inference(
        self,
        ort_outs: list,
        output_data: Any,
        module_name: str,
        output_names: list[str],
        verbose: bool = True,
    ):
        sqnr, mse, cosine_sim, _ = self._evaluate_outputs(
            ort_outs, output_data, module_name, output_names, verbose
        )
        if verbose:
            return ort_outs[-1]
        return ort_outs[-1], sqnr, mse, cosine_sim

    # ---------------------------------------------------------------------------
    # ONNX inference
    # ---------------------------------------------------------------------------

    def run_ort_inference(
        self, onnx_export_path: str, module_name: str, static_data: Any
    ) -> list:
        session    = self._build_ort_session(onnx_export_path)
        ort_inputs = self._build_ort_inputs(module_name, static_data)
        self._print_input_shapes(ort_inputs, self.logger)
        return session.run(None, ort_inputs)

    # ---------------------------------------------------------------------------
    # AIMET analysis
    # ---------------------------------------------------------------------------

    def build_aimet_callbacks(
        self,
    ) -> tuple[CallbackFunc, CallbackFunc, CallbackFunc, CallbackFunc]:
        """Wrap the four callback methods as AIMET CallbackFunc objects.

        Returns
        -------
        calib_forward, calib_eval  — backed by calib_dataloader (small subset).
        full_forward,  full_eval   — backed by full_dataloader  (entire dataset).
        """
        calib_forward = CallbackFunc(self.forward_pass_callback,      func_callback_args=None)
        calib_eval    = CallbackFunc(self.eval_callback,               func_callback_args=None)
        full_forward  = CallbackFunc(self.forward_pass_callback_full,  func_callback_args=None)
        full_eval     = CallbackFunc(self.eval_callback_full,          func_callback_args=None)
        return calib_forward, calib_eval, full_forward, full_eval

    def build_dummy_input(self) -> dict:
        """Return a single-sample input dict for the ONNX model.

        Used by QuantAnalyzer to trace the graph before analysis.
        Routes through ``_make_feed`` (via a throw-away session handle) so
        that MODULE_ORT_INPUT_ALIAS overrides are respected — e.g. the fused
        InternViT model needs a single ``{"input": images}`` feed, not the
        two-input ``{"images": ..., "inputs_embeds": ...}`` dict that
        ``_build_ort_inputs("InternViT", ...)`` would produce.
        """
        first_sample       = self.full_dataloader.dataset[0]["input"]
        first_sample_numpy = self._to_numpy(first_sample)
        # Build a temporary session solely to resolve the correct input names.
        tmp_session = self._build_ort_session(self.onnx_export_path)
        return self._make_feed(tmp_session, first_sample_numpy)

    def create_quantsim(
        self,
        onnx_model: Any,
        param_type: Any,
        activation_type: Any,
    ) -> QuantizationSimModel:
        """Instantiate and return a calibrated QuantizationSimModel.

        Uses post_training_tf scheme and htp_v81 config by default.
        """
        self.logger.debug(
            "Creating QuantizationSimModel (param=%s, activation=%s) providers=%s",
            param_type, activation_type, ORT_PROVIDERS,
        )
        return QuantizationSimModel(
            model=onnx_model,
            quant_scheme=QuantScheme.post_training_tf,
            config_file="htp_v81",
            param_type=param_type,
            activation_type=activation_type,
            providers=ORT_PROVIDERS,
        )

    def run_quant_analysis(
        self,
        onnx_model: Any,
        dummy_input: dict,
        qscheme: tuple[Any, Any],
        calib_forward: CallbackFunc,
        calib_eval: CallbackFunc,
        results_dir: str,
    ) -> None:
        """Run QuantAnalyzer.analyze() using the calibration dataset.

        Enables per-layer MSE loss analysis before running the full analysis.
        Results (sensitivity plots, MSE JSON, etc.) are written to results_dir.
        """
        self.logger.debug("Running QuantAnalyzer.analyze() ...")
        quant_analyzer = QuantAnalyzer(
            model=onnx_model,
            dummy_input=dummy_input,
            forward_pass_callback=calib_forward,
            eval_callback=calib_eval,
        )
        # MSE loss pass uses calib_dataloader (small subset, runs many times)
        quant_analyzer.enable_per_layer_mse_loss(
            unlabeled_dataset_iterable=self.calib_dataloader, num_batches=1
        )
        quant_analyzer.analyze(
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=qscheme[0],
            default_activation_bw=qscheme[1],
            config_file="htp_v73",
            results_dir=results_dir,
            providers=ORT_PROVIDERS,
        )
        self.logger.debug("analyze() complete. Results: %s", results_dir)

    def run_per_layer_sensitivity(
        self,
        sim: QuantizationSimModel,
        dummy_input: dict,
        full_forward: CallbackFunc,
        full_eval: CallbackFunc,
        mmp_results_dir: str,
    ) -> dict:
        """Run per-layer sensitivity analysis over the full dataset.

        Enables each quantizer one at a time and measures the eval metric drop.
        AIMET writes an HTML plot to mmp_results_dir and returns a
        ``{layer_name: eval_score}`` dict directly — no JSON file is produced.

        Returns the per-layer results dict.
        """
        self.logger.debug("Running perform_per_layer_analysis_by_enabling_quantizers() ...")
        # Fresh model copy required — QuantAnalyzer mutates the onnx.ModelProto in-place
        quant_analyzer_full = QuantAnalyzer(
            model=onnx.load(self.onnx_export_path),
            dummy_input=dummy_input,
            forward_pass_callback=full_forward,
            eval_callback=full_eval,
        )
        results = QuantAnalyzer.perform_per_layer_analysis_by_enabling_quantizers(
            quant_analyzer_full, sim, results_dir=mmp_results_dir
        )
        self.logger.debug("Per-layer analysis complete. Results: %s", mmp_results_dir)
        return results if results is not None else {}

    def _run_session_on_batch(
        self, session: InferenceSession, inputs: Any
    ) -> list:
        """Build the input feed from the session's own input names and run it.

        Zips session input names positionally against the provided tensors so
        this works for both plain ORT sessions and AIMET QuantSim sessions.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        input_names = [inp.name for inp in session.get_inputs()]
        if len(input_names) != len(inputs):
            raise ValueError(
                f"Session expects {len(input_names)} input(s) {input_names} "
                f"but got {len(inputs)} tensor(s)."
            )
        return session.run(None, dict(zip(input_names, inputs)))

    def forward_pass_callback(self, session: InferenceSession, num_batches: int | None = None) -> None:
        """Calibration forward pass — runs over calib_dataloader (small subset)."""
        self.logger.debug("Computing encodings with forward_pass_callback ...")
        for batch_idx, batch in enumerate(self.calib_dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            for idx in range(self._batch_size(batch)):
                inputs, _ = self._parse_batch_item(batch, idx)
                session.run(None, self._make_feed(session, inputs))

    def eval_callback(self, session: InferenceSession, num_batches: int | None = None) -> float:
        """Evaluation callback — runs over calib_dataloader (small subset).

        Returns mean SQNR (higher = better) so that
        perform_per_layer_analysis_by_enabling_quantizers correctly identifies
        sensitive layers as those with the LOWEST score.
        """
        mse_list, sqnr_list, cos_list = [], [], []
        for batch_idx, batch in enumerate(self.calib_dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            # Iterate sample by sample: ONNX models are exported with batch_size=1
            for idx in range(self._batch_size(batch)):
                inputs, outputs = self._parse_batch_item(batch, idx)
                ort_outs = session.run(None, self._make_feed(session, inputs))
                _, sqnr, mse, cosine_sim = self.verify_ort_inference(
                    ort_outs, outputs, self.module_name, self.output_names, verbose=False
                )
                mse_list.append(mse)
                sqnr_list.append(sqnr)
                cos_list.append(cosine_sim)
        return float(np.mean(sqnr_list))  # SQNR: higher = better (AIMET contract unchanged)

    def forward_pass_callback_full(self, session: InferenceSession, num_batches: int | None = None) -> None:
        """Forward pass over the full dataset — used by per-layer analysis."""
        for batch_idx, batch in enumerate(self.full_dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            for idx in range(self._batch_size(batch)):
                inputs, _ = self._parse_batch_item(batch, idx)
                session.run(None, self._make_feed(session, inputs))

    def eval_callback_full(self, session: InferenceSession, num_batches: int | None = None) -> float:
        """Evaluation callback over the full dataset — used by per-layer analysis.

        Returns mean SQNR (higher = better), consistent with eval_callback.
        """
        mse_list, sqnr_list, cos_list = [], [], []
        for batch_idx, batch in enumerate(self.full_dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            for idx in range(self._batch_size(batch)):
                inputs, outputs = self._parse_batch_item(batch, idx)
                ort_outs = session.run(None, self._make_feed(session, inputs))
                _, sqnr, mse, cosine_sim = self.verify_ort_inference(
                    ort_outs, outputs, self.module_name, self.output_names, verbose=False
                )
                mse_list.append(mse)
                sqnr_list.append(sqnr)
                cos_list.append(cosine_sim)
        return float(np.mean(sqnr_list))  # SQNR: higher = better (AIMET contract unchanged)

    def eval_full_dataset(
        self,
        onnx_path: str,
        label: str,
    ) -> dict[str, float]:
        """Run the ORT session at *onnx_path* over the entire full_dataloader.

        Accumulates per-sample SQNR, MSE, and cosine similarity, then returns
        their dataset-level means.  *label* is used only for logging
        (e.g. 'FP32', 'FP16', 'MMP').

        Returns
        -------
        dict with keys 'sqnr_db', 'mse', and 'cosine_sim'.
        """
        self.logger.debug("[%s] Evaluating over full dataset: %s", label, onnx_path)
        session = self._build_ort_session(onnx_path)
        mse_list, sqnr_list, cos_list = [], [], []
        per_output_lists: dict[str, dict[str, list]] = {
            name: {"sqnr": [], "mse": [], "cos": []} for name in self.output_names
        }
        for batch in self.full_dataloader:
            for idx in range(self._batch_size(batch)):
                inputs, outputs = self._parse_batch_item(batch, idx)
                ort_outs = session.run(None, self._make_feed(session, inputs))
                sqnr, mse, cosine_sim, per_output = self._evaluate_outputs(
                    ort_outs, outputs, self.module_name, self.output_names, verbose=False
                )
                mse_list.append(mse)
                sqnr_list.append(sqnr)
                cos_list.append(cosine_sim)
                for name, (s, m, c) in per_output.items():
                    per_output_lists[name]["sqnr"].append(s)
                    per_output_lists[name]["mse"].append(m)
                    per_output_lists[name]["cos"].append(c)
        avg_sqnr = float(np.mean(sqnr_list))
        avg_mse  = float(np.mean(mse_list))
        avg_cos  = float(np.mean(cos_list))
        self.logger.debug(
            "[%s] Dataset-level  SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
            label, avg_sqnr, avg_mse, avg_cos,
        )
        for name, lists in per_output_lists.items():
            if lists["sqnr"]:
                self.logger.debug(
                    "[%s]   %-12s SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
                    label, name,
                    float(np.mean(lists["sqnr"])),
                    float(np.mean(lists["mse"])),
                    float(np.mean(lists["cos"])),
                )
        return {"sqnr_db": avg_sqnr, "mse": avg_mse, "cosine_sim": avg_cos}

    def eval_sim(
        self,
        sim: QuantizationSimModel,
        label: str,
    ) -> dict[str, float]:
        """Evaluate *sim.session* (the live quantized session) over full_dataloader.

        Unlike eval_full_dataset, this runs inference directly through the
        QuantizationSimModel's in-memory session so the quantized weights and
        encodings that were just calibrated are what actually get measured —
        no file round-trip, no risk of loading the wrong ONNX.

        Returns
        -------
        dict with keys 'sqnr_db', 'mse', and 'cosine_sim'.
        """
        self.logger.debug("[%s] Evaluating sim.session over full dataset", label)
        session = sim.session
        mse_list, sqnr_list, cos_list = [], [], []
        per_output_lists: dict[str, dict[str, list]] = {
            name: {"sqnr": [], "mse": [], "cos": []} for name in self.output_names
        }
        for batch in self.full_dataloader:
            for idx in range(self._batch_size(batch)):
                inputs, outputs = self._parse_batch_item(batch, idx)
                ort_outs = session.run(None, self._make_feed(session, inputs))
                sqnr, mse, cosine_sim, per_output = self._evaluate_outputs(
                    ort_outs, outputs, self.module_name, self.output_names, verbose=False
                )
                mse_list.append(mse)
                sqnr_list.append(sqnr)
                cos_list.append(cosine_sim)
                for name, (s, m, c) in per_output.items():
                    per_output_lists[name]["sqnr"].append(s)
                    per_output_lists[name]["mse"].append(m)
                    per_output_lists[name]["cos"].append(c)
        avg_sqnr = float(np.mean(sqnr_list))
        avg_mse  = float(np.mean(mse_list))
        avg_cos  = float(np.mean(cos_list))
        self.logger.debug(
            "[%s] Dataset-level  SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
            label, avg_sqnr, avg_mse, avg_cos,
        )
        for name, lists in per_output_lists.items():
            if lists["sqnr"]:
                self.logger.debug(
                    "[%s]   %-12s SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
                    label, name,
                    float(np.mean(lists["sqnr"])),
                    float(np.mean(lists["mse"])),
                    float(np.mean(lists["cos"])),
                )
        return {"sqnr_db": avg_sqnr, "mse": avg_mse, "cosine_sim": avg_cos}

# ---------------------------------------------------------------------------
# Quantization scheme candidates & fallback chain
# ---------------------------------------------------------------------------

# Ordered from highest to lowest precision — used to drive the qscheme loop
QSCHEME_ORDER = ["fp32", "fp16", "w8a16", "w8a8", "w4a16", "w4a4"]

# param_type, activation_type for each quantized scheme.
# fp32 is evaluated as a plain ORT baseline (no QuantSim needed).
# fp16 is the target-device reference — all lower-precision qschemes are
# compared against it for MMP decisions.
QSCHEMES = {
    "fp32":  None,                                           # plain ORT — original model, informational only
    "fp16":  (aimet_onnx.float16, aimet_onnx.float16),      # target-device reference
    "w8a16": (aimet_onnx.int8,    aimet_onnx.int16),
    "w8a8":  (aimet_onnx.int8,    aimet_onnx.int8),
    "w4a16": (aimet_onnx.int4,    aimet_onnx.int16),
    "w4a4":  (aimet_onnx.int4,    aimet_onnx.int4),
}

# ---------------------------------------------------------------------------
# MMP fallback chains
# ---------------------------------------------------------------------------
# FP32 is NOT available on the target device and is therefore NEVER used as
# a fallback precision.  FP16 is the highest precision available on-device
# and serves as both the quality reference and the ultimate MMP fallback.
#
# Precision preference order (best quality → least):  fp16 > w8a16 > w8a8
#
# Each qscheme maps to an ordered list of escalation steps.  MMP runs one
# sensitivity scan per step, flipping only the layers that are STILL
# sensitive at the current precision.  Already-promoted layers are never
# re-scanned.
#
# Step encoding:
#   (QuantizationDataType, bw)
#       → promote ALL quantizers of the layer to that precision
#   {"param": (QuantizationDataType, bw), "act": (QuantizationDataType, bw)}
#       → promote param and activation quantizers independently
#         (used for w8a16: keep weights at int8, raise activations to int16)
#
# Chains:
#   fp16  → []                  no MMP (fp16 IS the reference)
#   w8a16 → [fp16]              1 scan:  w8a16-sensitive layers → fp16
#   w8a8  → [w8a16, fp16]       2 scans: w8a8-sensitive → w8a16 (step 1),
#                                        still-sensitive → fp16  (step 2)
#   w4a16 → [w8a16, fp16]       2 scans: w4a16-sensitive → w8a16 (step 1),
#                                        still-sensitive → fp16  (step 2)
FALLBACK_CHAINS: dict[str, list] = {
    # fp16 is the on-device reference — no further fallback available
    "fp16":  [],
    # w8a16: sensitive layers promoted to fp16 (single step)
    "w8a16": [
        (QuantizationDataType.float, 16),                                           # step 1 (mmp): → fp16
    ],
    # w8a8:
    #   step 1 (v1): sensitive layers → w8a16
    #   step 2 (v2): still-sensitive layers split by epsilon:
    #                  score in [threshold-epsilon, threshold) → w8a16 (low tier)
    #                  score < threshold-epsilon               → fp16  (high tier)
    "w8a8":  [
        {"param": (QuantizationDataType.int, 8), "act": (QuantizationDataType.int, 16)},   # step 1 (v1): → w8a16
        [
            {"param": (QuantizationDataType.int, 8), "act": (QuantizationDataType.int, 16)},  # tier_low:  w8a16
            (QuantizationDataType.float, 16),                                                  # tier_high: fp16
        ],                                                                                     # step 2 (v2): two-tier
    ],
    # w4a16:
    #   step 1 (v1): sensitive layers → w8a16
    #   step 2 (v2): still-sensitive layers split by epsilon (same as w8a8)
    "w4a16": [
        {"param": (QuantizationDataType.int, 8), "act": (QuantizationDataType.int, 16)},   # step 1 (v1): → w8a16
        [
            {"param": (QuantizationDataType.int, 8), "act": (QuantizationDataType.int, 16)},  # tier_low:  w8a16
            (QuantizationDataType.float, 16),                                                  # tier_high: fp16
        ],                                                                                     # step 2 (v2): two-tier
    ],
}


def _fallback_label(step) -> str:
    """Human-readable label for a single fallback chain step.

    Handles four step encodings:
      None                                         → 'disabled (fp32)'  [legacy]
      (QuantizationDataType, bw)                   → 'fp{bw}' or 'int{bw}'
      {"param": (dtype, bw), "act": (dtype, bw)}   → 'w{p_bw}a{a_bw}'  e.g. 'w8a16'
      [tier_low, tier_high]                        → '{label(tier_low)}|{label(tier_high)}'
    """
    if step is None:
        return "disabled (fp32)"
    if isinstance(step, list):
        return f"{_fallback_label(step[0])}|{_fallback_label(step[1])}"
    if isinstance(step, dict):
        _, p_bw = step["param"]
        _, a_bw = step["act"]
        return f"w{p_bw}a{a_bw}"   # e.g. "w8a16"
    dtype, bw = step
    return f"fp{bw}" if dtype == QuantizationDataType.float else f"int{bw}"


# Negligible degradation threshold: if |quant_sqnr - fp16_ref_sqnr| < this, skip MMP.
# Anchored to the fp16 reference (not fp32) since fp32 is not available on device.
MMP_SKIP_SQNR_DELTA_DB: float = 0.5

MMP_SENSITIVITY_MARGIN_DB: float = 0.0

# Two-tier epsilon split (used in escalation steps with [tier_low, tier_high]):
#   score in [threshold - epsilon, threshold) → promote to tier_low  (e.g. w8a16)
#   score < threshold - epsilon               → promote to tier_high (e.g. fp16)
MMP_TIER_SPLIT_EPSILON_DB: float = 1.5


def _fmt(m: dict | None) -> str:
    """Format a metrics dict as 'SQNR / MSE / CosSim' for the summary table."""
    if m is None:
        return "     —     /     —     /     —    "
    cos = m.get("cosine_sim")
    cos_str = f"{cos:.4f}" if cos is not None else "  —   "
    return f"{m['sqnr_db']:7.2f} dB / {m['mse']:.3e} / {cos_str}"


def print_results_table(
    module_name: str,
    orig_key: str,
    orig_metrics: dict,
    rows: list[dict],
    logger: logging.Logger,
    fp32_info_metrics: dict | None = None,
) -> None:
    """Print the per-module quantization results summary table.

    Parameters
    ----------
    module_name       : display name for the module (may include opt_tag).
    orig_key          : label for the reference row (e.g. 'fp16').
    orig_metrics      : metrics for the reference row.
    rows              : list of dicts with keys 'qscheme', 'quant_metrics', 'mmp_metrics'.
    logger            : logger to write the table to.
    fp32_info_metrics : optional FP32 metrics shown as an informational row
                        (not on target device — not used as MMP reference).
    """
    col_w = 38  # wider to fit SQNR / MSE / CosSim
    sep   = "─" * (10 + col_w * 2 + 4)

    lines = [
        f"┌{sep}┐",
        f"│ Module: {module_name:<{len(sep) - 10}}│",
        f"├{'─'*10}┬{'─'*col_w}┬{'─'*col_w}┤",
        f"│{'QScheme':<10}│{'Quantization':<{col_w}}│{'MMP':<{col_w}}│",
        f"│{'':10}│{'SQNR (dB) / MSE / CosSim':<{col_w}}│{'SQNR (dB) / MSE / CosSim':<{col_w}}│",
        f"├{'─'*10}┼{'─'*col_w}┼{'─'*col_w}┤",
    ]
    # Optional FP32 informational row — not available on target device
    if fp32_info_metrics is not None:
        lines.append(
            f"│{'fp32 ℹ':<10}│{_fmt(fp32_info_metrics):<{col_w}}│{'— (not on device)':<{col_w}}│"
        )
        lines.append(f"├{'─'*10}┼{'─'*col_w}┼{'─'*col_w}┤")
    # Reference row (fp16 ◀) — the on-device quality anchor
    lines.append(
        f"│{(orig_key + ' ◀'):<10}│{_fmt(orig_metrics):<{col_w}}│{'— (reference)':<{col_w}}│"
    )
    lines.append(f"├{'─'*10}┼{'─'*col_w}┼{'─'*col_w}┤")
    for row in rows:
        quant_str = _fmt(row["quant_metrics"])
        mmp_str   = _fmt(row["mmp_metrics"])
        lines.append(
            f"│{row['qscheme']:<10}│{quant_str:<{col_w}}│{mmp_str:<{col_w}}│"
        )
    lines.append(f"└{'─'*10}┴{'─'*col_w}┴{'─'*col_w}┘")

    table_str = "\n".join(lines)
    logger.info("\n%s", table_str)


def print_mmp_layer_table(
    qscheme_key: str,
    sensitivity_data: dict[str, float],
    htp_disabled_layers: list[str],
    flipped_layers: list[tuple[str, float, str]],
    remaining_layers: list[tuple[str, float]],
    logger: logging.Logger,
) -> None:
    """Log a per-layer MMP breakdown table with three sections:

    1. Disabled by htp_v81 config  — quantizers already off before MMP.
    2. Flipped by MMP chain        — sensitive layers and the exact precision
                                     each ended up at (may differ per layer
                                     when the chain has multiple steps).
    3. Remaining at qscheme        — insensitive layers kept at base precision.

    *flipped_layers* is a list of (layer_name, sqnr, accepted_step_label).
    *sensitivity_data* maps layer_name → SQNR from the initial scan; layers
    in htp_disabled_layers have no score so their SQNR column shows '—'.
    """
    layer_col  = 42
    sqnr_col   = 12
    status_col = 28

    def _row(layer: str, sqnr: float | None, status: str) -> str:
        sqnr_str = f"{sqnr:8.2f} dB" if sqnr is not None else f"{'—':^{sqnr_col}}"
        return f"│ {layer:<{layer_col}} │ {sqnr_str:>{sqnr_col}} │ {status:<{status_col}} │"

    section_sep = f"├{'─'*(layer_col+2)}┼{'─'*(sqnr_col+2)}┼{'─'*(status_col+2)}┤"
    top    = f"┌{'─'*(layer_col+2)}┬{'─'*(sqnr_col+2)}┬{'─'*(status_col+2)}┐"
    bottom = f"└{'─'*(layer_col+2)}┴{'─'*(sqnr_col+2)}┴{'─'*(status_col+2)}┘"

    lines = [
        top,
        f"│ {'Layer':<{layer_col}} │ {'SQNR':>{sqnr_col}} │ {'Precision / Status':<{status_col}} │",
        section_sep,
    ]

    # ── Section 1: disabled by htp_v81 ────────────────────────────────────────
    if htp_disabled_layers:
        lines.append(
            f"│ {'── disabled by htp_v81 config ──':<{layer_col}} "
            f"│ {'':>{sqnr_col}} │ {'':>{status_col}} │"
        )
        for layer in sorted(htp_disabled_layers):
            lines.append(_row(layer, None, "disabled (htp_v81)"))
        lines.append(section_sep)

    # ── Section 2: flipped by MMP chain ─────────────────────────────────────
    if flipped_layers:
        lines.append(
            f"│ {'── flipped by MMP chain ──':<{layer_col}} "
            f"│ {'':>{sqnr_col}} │ {'':>{status_col}} │"
        )
        # Sort worst SQNR first so the most sensitive layers appear at the top.
        for layer, sqnr, step_label in sorted(flipped_layers, key=lambda x: x[1]):
            status = f"{qscheme_key} → {step_label}"
            lines.append(_row(layer, sqnr, status))
        lines.append(section_sep)

    # ── Section 3: remaining at base qscheme ────────────────────────────────
    if remaining_layers:
        lines.append(
            f"│ {'── remaining at ' + qscheme_key + ' ──':<{layer_col}} "
            f"│ {'':>{sqnr_col}} │ {'':>{status_col}} │"
        )
        for layer, sqnr in sorted(remaining_layers, key=lambda x: x[1]):
            lines.append(_row(layer, sqnr, qscheme_key))

    lines.append(bottom)
    logger.debug("MMP per-layer breakdown:\n%s", "\n".join(lines))


# ---------------------------------------------------------------------------
# Job configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ONNX module name → dataloader/callback alias
# ---------------------------------------------------------------------------
# The ONNX file names are long, descriptive identifiers (e.g. the full fused
# module name including optimisation tag).  The dataloader, _build_ort_inputs,
# output_names logic, and eval_metrics all use short canonical aliases that
# match the keys in _MODULE_PT_FILE_MAP and the existing per-module branches.
#
# Add an entry here whenever a new fused ONNX name is introduced so that
# quant.py can resolve the correct dataloader alias without touching any other
# code path.
MODULE_ALIAS: dict[str, str] = {
    # fused ONNX module name                              → canonical alias
    "InternViT300M_Pixel_Unshuffle_MLP1":                  "InternViT",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA":               "Qwen2_05_VLM",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_katana": "Qwen2_05_VLM",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_sabre":  "Qwen2_05_VLM",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_oob":    "Qwen2_05_VLM",
    # modules whose ONNX name already matches the alias need no entry here
}

# Per-module ORT input-mapping alias.
#
# MODULE_ALIAS drives the *dataloader* (which .pt files to load) and
# output_names.  But the fused ONNX models may have a different input
# interface than the unfused canonical module.
#
# Example: InternViT300M_Pixel_Unshuffle_MLP1 aliases to "InternViT" for
# the dataloader (same vit.pt files), but the fused ONNX has a single
# input named "input" — not the two-input {"images", "inputs_embeds"}
# interface of the standalone InternViT.onnx.  Passing it through
# _build_ort_inputs("InternViT", ...) would produce the wrong feed dict.
#
# Entries here override which key is used in _build_ort_inputs.
# Use None to force the generic single-input passthrough.
# Modules not listed fall back to MODULE_ALIAS (or module_name if no alias).
MODULE_ORT_INPUT_ALIAS: dict[str, str | None] = {
    # Keyed on the canonical alias (= self.module_name inside Quanter).
    # fused ONNX module alias  → _build_ort_inputs key  (None = generic passthrough)
    "InternViT": None,   # InternViT300M_Pixel_Unshuffle_MLP1 has a single input named "input"
}

# Per-module ONNX filename suffix overrides.
#
# Problem: some module names already embed the optimisation tag (e.g.
# "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_katana"), so
# appending the global ONNX_SUFFIX[optimized] ("_Piqaro_katana_v2.onnx")
# would produce a doubled suffix:
#   ...Piqaro_katana_Piqaro_katana_v2.onnx  ← wrong
#
# For these modules the suffix should be just "_v2.onnx" / "_v1.onnx" /
# "_v0.onnx" because the opt-tag is already baked into the module name.
#
# Layout:  module_name → {optimized_int: suffix_string}
# Modules not listed here use the global ONNX_SUFFIX table unchanged.
MODULE_ONNX_SUFFIX: dict[str, str] = {
    # These module names already embed the opt-tag, so the suffix is a fixed
    # version stamp — the same regardless of --optimized.  The opt-tag in the
    # module name IS the variant selector; --optimized only controls which
    # module name is used, not a different version of the same file.
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_katana": "_v2.onnx",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_sabre":  "_v1.onnx",
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_oob":    "_v0.onnx",
}


# Default module list — used when running quant.py directly (no --module flag).
# When launched via launch_quant.py each subprocess receives exactly one module
# via --module and this list is ignored.
DEFAULT_MODULES: list[str] = [
    # "vit",
    # "InternViT",
    # "InternViT_Interleaving",
    # "pixel_unshuffle",
    # "non_CoT_token_interleaver",
    # "llm_ar32",
    # "InternViT300M_Pixel_Unshuffle_MLP1",
    # "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA_Piqaro_katana",
    "mlp1",
]

# qschemes to evaluate per module.
# fp16 MUST be first — it runs as the on-device reference before any lower-
# precision qscheme, and its metrics are captured as reference_metrics for
# all subsequent MMP decisions.
MODULE_ACTIVE_QSCHEMES: dict[str, list[str]] = {
    "InternViT300M_Pixel_Unshuffle_MLP1":             ["fp16", "w8a16", "w8a8"],
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA": ["fp16", "w8a16", "w4a16"],
}
DEFAULT_ACTIVE_QSCHEMES = ["fp16", "w8a16", "w8a8"]   # fallback for any unlisted module

DATA_PATH = "/local/mnt2/workspace2/users/sathya/dataset/SimLingo/Closed-Loop/Bench2Drive220/Intermediate_Results/without-CoT/simlingo_b2d_traj-FP32/debug_viz/simlingo/iter_013.ckpt/leaderboard/bench2drive220_0_simlingo_traj_2026_02_10_15_40_34/intermediate_data/"

OPT_TAGS    = {0: "oob",   1: "sabre",                    2: "katana"}
ONNX_SUFFIX = {0: "_v0.onnx", 1: "_Piqaro_sabre_v1.onnx", 2: "_Piqaro_katana_v2.onnx"}
ONNX_BASE   = "/local/mnt/workspace/users/sathya/projects/simlingo/quantization/onnx/fused"
RESULTS_BASE = "/local/mnt/workspace/users/sathya/projects/simlingo/quantization/quant_analyzer_results"

# Resolve: CLI --optimized wins over the hard-coded default below.
_DEFAULT_OPTIMIZED = 0   # 0: OOB, 1: Sabre, 2: Katana  ← edit this for interactive runs
optimized = _ARGS.optimized if _ARGS.optimized is not None else _DEFAULT_OPTIMIZED
opt_tag   = OPT_TAGS[optimized]   # "oob" | "sabre" | "katana"

# Resolve module list: single --module from CLI, or the full DEFAULT_MODULES list.
modules: list[str] = [_ARGS.module] if _ARGS.module is not None else DEFAULT_MODULES


if __name__ == "__main__":
    for module_name in tqdm(modules, desc="Modules"):
        # ------------------------------------------------------------------
        # Per-module setup: dataloaders, paths, baselines
        # ------------------------------------------------------------------
        # fp16 is the on-device reference; fp32 is informational only.
        orig_key = "fp16"
    
        # Resolve the canonical alias used by the dataloader, output_names, and
        # _build_ort_inputs.  Falls back to module_name itself when no alias is
        # registered (i.e. the ONNX name already matches the canonical name).
        module_alias = MODULE_ALIAS.get(module_name, module_name)
    
        # run_key scopes every output (logs, results, exports) under opt_tag so
        # oob / sabre / katana runs never overwrite each other.
        run_key = f"{module_name}/{opt_tag}"   # e.g. "mlp1/sabre"
    
        logger = setup_logger(run_key, "summary")
        if module_alias != module_name:
            logger.info("Module alias: %s → %s (dataloader/callbacks)", module_name, module_alias)
    
        # Also route AIMET's own loggers into the same module log file so
        # lines like "Selecting DefaultOpInstanceConfigGenerator" are captured.
        fh = _get_module_file_handler(run_key)
        for aimet_logger_name in ("Quant", "QuantAnalyzer", "lightning_fabric"):
            al = logging.getLogger(aimet_logger_name)
            if fh not in al.handlers:
                al.addHandler(fh)
    
        # Resolve the correct ONNX filename suffix for this module.
        # Modules in MODULE_ONNX_SUFFIX already embed the opt-tag in their name,
        # so they use a fixed version suffix regardless of --optimized.
        # All other modules use the global ONNX_SUFFIX table keyed by --optimized.
        if module_name in MODULE_ONNX_SUFFIX:
            _onnx_suffix = MODULE_ONNX_SUFFIX[module_name]
        else:
            _onnx_suffix = ONNX_SUFFIX[optimized]
        onnx_export_path = f"{ONNX_BASE}/{module_name}{_onnx_suffix}"
        logger.debug("ONNX path: %s", onnx_export_path)
    
        # output_names and dataloader use the canonical alias, not the ONNX name
        output_names = ["route", "speed"] if module_alias in ("llm_ar32", "Qwen2_05_VLM") else ["output"]
    
        dataloader, calib_dataloader = Quanter.build_dataloaders(
            data_path=DATA_PATH,
            module_name=module_alias,   # ← alias: matches .pt file names in the dataset
        )
        logger.debug("Full dataset size : %d", len(dataloader.dataset))
        logger.debug("Calib dataset size: %d", len(calib_dataloader.dataset))
    
        # Shared Quanter instance — reused across all qschemes for this module.
        # module_alias is passed so that _build_ort_inputs and output_names logic
        # resolve correctly against the canonical per-module branches.
        quanter = Quanter(
            onnx_export_path,
            module_alias,              # ← alias: drives _build_ort_inputs / output_names
            static_data=None,
            calib_dataloader=calib_dataloader,
            full_dataloader=dataloader,
            output_names=output_names,
            logger=logger,
        )
    
        # results_dir is scoped by both module and opt_tag
        # e.g. quant_analyzer_results/mlp1/sabre/
        results_dir = os.path.join(RESULTS_BASE, module_name, opt_tag)
        os.makedirs(results_dir, exist_ok=True)
    
        # ------------------------------------------------------------------
        # Step 1: FP32 evaluation — informational only.
        # FP32 is NOT available on the target device and is NOT used as the
        # MMP reference.  We log its metrics purely for offline comparison.
        # ------------------------------------------------------------------
        fp32_metrics = quanter.eval_full_dataset(onnx_export_path, label="FP32 [info]")
        logger.info(
            "[FP32 info] SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f  "
            "(not on device — informational only)",
            fp32_metrics["sqnr_db"], fp32_metrics["mse"], fp32_metrics["cosine_sim"],
        )
    
        # Resolve per-module qscheme list (fp16 must always be first).
        active_qschemes = MODULE_ACTIVE_QSCHEMES.get(module_name, DEFAULT_ACTIVE_QSCHEMES)
    
        # reference_metrics is set after fp16 runs (first in active_qschemes).
        # All MMP decisions for w8a16 / w8a8 / ... are anchored to this value.
        reference_metrics: dict | None = None
    
        # Accumulate one row per qscheme (excluding fp16, which is the reference row)
        table_rows: list[dict] = []
        all_results: dict = {
            "module":    module_name,
            "alias":     module_alias,
            "opt":       opt_tag,
            "fp32_info": fp32_metrics,   # informational — not the MMP anchor
            "reference": "fp16",
        }
    
        # ------------------------------------------------------------------
        # Per-qscheme loop
        # ------------------------------------------------------------------
        for qscheme_key in active_qschemes:
            qs_logger = setup_logger(run_key, qscheme_key)
            qs_logger.debug("Module: %s | opt: %s | QScheme: %s", module_name, opt_tag, qscheme_key)
            quanter.logger = qs_logger   # redirect quanter logs to the per-qscheme logger
    
            qscheme_results_dir = os.path.join(results_dir, qscheme_key)
            mmp_results_dir     = os.path.join(qscheme_results_dir, "mmp")
            os.makedirs(mmp_results_dir, exist_ok=True)
    
            quant_metrics = None
            mmp_metrics   = None
    
            # All active qschemes go through the full QuantSim path.
            # fp32 is never in ACTIVE_QSCHEMES so QSCHEMES[qscheme_key] is always non-None here.
            param_type, activation_type = QSCHEMES[qscheme_key]
            onnx_model_for_sim = onnx.load(onnx_export_path)
    
            sim = quanter.create_quantsim(onnx_model_for_sim, param_type, activation_type)
            sim.compute_encodings(quanter.forward_pass_callback)
    
            # Export the calibrated quantized model and encodings unconditionally —
            # this is the primary deliverable regardless of whether MMP is needed.
            export_prefix = f"{module_name}_{qscheme_key}_quantized"
            sim.export(path=qscheme_results_dir, filename_prefix=export_prefix, export_model=True)
            qs_logger.debug("Exported quantized model: %s/%s.onnx", qscheme_results_dir, export_prefix)
    
            # Evaluate directly through sim.session — no file round-trip
            quant_metrics = quanter.eval_sim(sim, label=qscheme_key)
    
            # ------------------------------------------------------------------
            # Step 2: fp16 is the on-device reference — capture its metrics and
            # skip MMP (it has no fallback chain).
            # ------------------------------------------------------------------
            if qscheme_key == "fp16":
                reference_metrics = quant_metrics
                qs_logger.info(
                    "[FP16 reference] SQNR: %.2f dB | MSE: %.6e | CosSim: %.6f",
                    reference_metrics["sqnr_db"], reference_metrics["mse"],
                    reference_metrics["cosine_sim"],
                )
                all_results["fp16"] = {"quant": quant_metrics}
                # fp16 is shown as the reference row in the table, not a data row
                del sim
                continue
    
            # ------------------------------------------------------------------
            # MMP decision — anchored to fp16 reference (not fp32).
            # ------------------------------------------------------------------
            if reference_metrics is None:
                qs_logger.warning(
                    "reference_metrics not set (fp16 must run before %s) — skipping MMP.",
                    qscheme_key,
                )
                table_rows.append({
                    "qscheme":       qscheme_key,
                    "quant_metrics": quant_metrics,
                    "mmp_metrics":   None,
                })
                all_results[qscheme_key] = {"reference": reference_metrics, "quant": quant_metrics, "mmp": None}
                del sim
                continue
    
            fallback_chain = FALLBACK_CHAINS.get(qscheme_key, [])
    
            if not fallback_chain:
                # No MMP chain defined for this qscheme
                qs_logger.debug("No MMP chain defined for %s — skipping MMP.", qscheme_key)
            else:
                sqnr_delta = abs(quant_metrics["sqnr_db"] - reference_metrics["sqnr_db"])
                if sqnr_delta < MMP_SKIP_SQNR_DELTA_DB:
                    qs_logger.debug(
                        "SQNR delta %.2f dB < %.1f dB threshold vs fp16 reference — skipping MMP.",
                        sqnr_delta, MMP_SKIP_SQNR_DELTA_DB,
                    )
                else:
                    calib_forward, calib_eval, full_forward, full_eval = quanter.build_aimet_callbacks()
                    dummy_input = quanter.build_dummy_input()
    
                    # ------------------------------------------------------------------
                    # Sensitivity-driven MMP: single enabling scan per chain step.
                    #
                    # Oracle: perform_per_layer_analysis_by_enabling_quantizers
                    # Threshold: fp16_ref_sqnr − MMP_SENSITIVITY_MARGIN_DB
                    #
                    # For each step in the fallback chain:
                    #   1. Run enabling analysis on the current (possibly mixed) sim.
                    #   2. Flip ALL layers whose score < threshold AND whose current
                    #      precision == candidate_precision for this step.
                    #   3. Re-compute encodings; evaluate.
                    #   4. If delta < MMP_SKIP_SQNR_DELTA_DB → converged, stop.
                    #
                    # Two-tier steps (isinstance(step, list)):
                    #   score in [threshold-epsilon, threshold) → tier_low  (e.g. w8a16)
                    #   score < threshold-epsilon               → tier_high (e.g. fp16)
                    # ------------------------------------------------------------------
    
                    # Build op_name → quantizer maps
                    op_name_to_quantizers:       dict[str, list] = {}
                    op_name_to_param_quantizers: dict[str, list] = {}
                    op_name_to_act_quantizers:   dict[str, list] = {}
                    for op in sim.connected_graph.ordered_ops:
                        in_qs, out_qs, param_qs = sim.get_op_quantizers(op)
                        act_qs_list   = list(in_qs) + list(out_qs)
                        param_qs_list = list(param_qs.values())
                        all_qs        = act_qs_list + param_qs_list
                        if all_qs:
                            op_name_to_quantizers[op.name_op]       = all_qs
                            op_name_to_param_quantizers[op.name_op] = param_qs_list
                            op_name_to_act_quantizers[op.name_op]   = act_qs_list
    
                    # Snapshot quantizer state BEFORE any MMP flips.
                    total_q          = len(sim.qc_quantize_op_dict)
                    pre_flip_enabled = sum(1 for q in sim.qc_quantize_op_dict.values() if q.enabled)
                    n_htp_disabled   = total_q - pre_flip_enabled
                    pre_disabled_ids: set = {
                        id(q) for q in sim.qc_quantize_op_dict.values() if not q.enabled
                    }
    
                    # Classify ops: htp_disabled (all qs off by config) vs active.
                    htp_disabled_layers: list[str] = []
                    active_ops: list[str] = []
                    for op_name, qs in op_name_to_quantizers.items():
                        if all(id(q) in pre_disabled_ids for q in qs):
                            htp_disabled_layers.append(op_name)
                        else:
                            active_ops.append(op_name)
    
                    layer_precision: dict[str, str] = {name: qscheme_key for name in active_ops}
                    # sensitivity_threshold is computed per-step after the scan (see below).
                    sensitivity_threshold: float = 0.0
    
                    from aimet_onnx.qc_quantize_op import OpMode
    
                    def _apply_step_to_layer(layer_name: str, step) -> int:
                        """Apply *step* to all active quantizers of *layer_name*.
    
                        step can be:
                          (QuantizationDataType, bw)
                              → promote ALL quantizers to that precision
                          {"param": (dtype, bw), "act": (dtype, bw)}
                              → promote param and activation quantizers independently
                          None
                              → disable quantizer (fp32 passthrough — legacy)
    
                        Returns the number of quantizers actually changed.
                        """
                        changed = 0
                        if isinstance(step, dict):
                            param_step = step.get("param")
                            act_step   = step.get("act")
                            for q in op_name_to_param_quantizers.get(layer_name, []):
                                if id(q) in pre_disabled_ids:
                                    continue
                                if param_step is None:
                                    q.enabled = False
                                else:
                                    fb_dtype, fb_bw = param_step
                                    q.enabled   = True
                                    q.data_type = fb_dtype
                                    q.set_bitwidth(fb_bw)
                                    q.op_mode   = OpMode.quantizeDequantize
                                changed += 1
                            for q in op_name_to_act_quantizers.get(layer_name, []):
                                if id(q) in pre_disabled_ids:
                                    continue
                                if act_step is None:
                                    q.enabled = False
                                else:
                                    fb_dtype, fb_bw = act_step
                                    q.enabled   = True
                                    q.data_type = fb_dtype
                                    q.set_bitwidth(fb_bw)
                                    q.op_mode   = OpMode.quantizeDequantize
                                changed += 1
                        else:
                            for q in op_name_to_quantizers.get(layer_name, []):
                                if id(q) in pre_disabled_ids:
                                    continue
                                if step is None:
                                    q.enabled = False
                                else:
                                    fb_dtype, fb_bw = step
                                    q.enabled   = True
                                    q.data_type = fb_dtype
                                    q.set_bitwidth(fb_bw)
                                    q.op_mode   = OpMode.quantizeDequantize
                                changed += 1
                        return changed
    
                    def _ensure_qdq_mode() -> None:
                        for q in sim.qc_quantize_op_dict.values():
                            if q.enabled:
                                q.op_mode = OpMode.quantizeDequantize
    
                    # sensitivity_data from the last step that ran (initialized empty
                    # so the final outcome lists work even if no step ran).
                    sensitivity_data: dict[str, float] = {}
    
                    step_converged = False
                    for step_idx, step in enumerate(fallback_chain):
                        step_num    = step_idx + 1
                        is_two_tier = isinstance(step, list)
                        step_label  = _fallback_label(step)
    
                        # Determine candidate precision for this step:
                        #   step 1 → layers still at the base qscheme
                        #   step 2+ → layers promoted in the PREVIOUS step
                        #             (for two-tier prev steps, use the low-tier label
                        #              since those are the ones that may still need escalation)
                        if step_idx == 0:
                            candidate_precision = qscheme_key
                        else:
                            prev_step = fallback_chain[step_idx - 1]
                            if isinstance(prev_step, list):
                                candidate_precision = _fallback_label(prev_step[0])
                            else:
                                candidate_precision = _fallback_label(prev_step)
    
                        qs_logger.debug(
                            "MMP step-%d (%s → %s): running enabling analysis ...",
                            step_num, candidate_precision, step_label,
                        )
    
                        # Run enabling analysis on the current (possibly mixed) sim.
                        step_mmp_dir = os.path.join(mmp_results_dir, f"step{step_num}")
                        os.makedirs(step_mmp_dir, exist_ok=True)
    
                        # ── Parallel or sequential sensitivity scan ──────────────
                        _use_parallel = (
                            _ARGS.parallel_sensitivity
                            and _ARGS.sensitivity_gpu_ids is not None
                        )
                        if _use_parallel:
                            _sens_gpu_ids = [
                                int(g.strip())
                                for g in _ARGS.sensitivity_gpu_ids.split(",")
                                if g.strip()
                            ]
                            _sn_per_worker  = getattr(_ARGS, "sn_per_worker", 0.0)
                            _use_live_poll  = not getattr(_ARGS, "no_live_vram_poll", False)
                            _use_dynamic    = getattr(_ARGS, "dynamic_scheduling", False)
                            _gpu_headroom   = getattr(_ARGS, "gpu_headroom_gb", 1.0)
    
                            qs_logger.info(
                                "MMP step-%d: using %s sensitivity scan "
                                "(gpu_ids=%s, vram_per_gpu=%.0f GB, cushion=%.0f GB, "
                                "sn_per_worker=%.2f GB, gpu_headroom=%.2f GB, live_poll=%s)",
                                step_num,
                                "dynamic Ray-based" if _use_dynamic else "parallel subprocess",
                                _sens_gpu_ids,
                                _ARGS.vram_per_gpu, _ARGS.cushion_gb,
                                _sn_per_worker, _gpu_headroom, _use_live_poll,
                            )
    
                            if _use_dynamic:
                                from dynamic_scheduler import run_parallel_sensitivity_dynamic
                                sensitivity_data = run_parallel_sensitivity_dynamic(
                                    quanter=quanter,
                                    sim=sim,
                                    dummy_input=dummy_input,
                                    full_forward=full_forward,
                                    full_eval=full_eval,
                                    mmp_results_dir=step_mmp_dir,
                                    gpu_ids=_sens_gpu_ids,
                                    vram_per_gpu_gb=_ARGS.vram_per_gpu,
                                    cushion_gb=_ARGS.cushion_gb,
                                    qscheme_key=qscheme_key,
                                    op_name_to_quantizers=op_name_to_quantizers,
                                    htp_disabled_layers=htp_disabled_layers,
                                    data_path=DATA_PATH,
                                    step_num=step_num,
                                    sn_per_worker_gb=_sn_per_worker,
                                    gpu_headroom_gb=_gpu_headroom,
                                )
                            else:
                                from parallel_sensitivity import run_parallel_sensitivity
                                sensitivity_data = run_parallel_sensitivity(
                                    quanter=quanter,
                                    sim=sim,
                                    dummy_input=dummy_input,
                                    full_forward=full_forward,
                                    full_eval=full_eval,
                                    mmp_results_dir=step_mmp_dir,
                                    gpu_ids=_sens_gpu_ids,
                                    vram_per_gpu_gb=_ARGS.vram_per_gpu,
                                    cushion_gb=_ARGS.cushion_gb,
                                    qscheme_key=qscheme_key,
                                    op_name_to_quantizers=op_name_to_quantizers,
                                    htp_disabled_layers=htp_disabled_layers,
                                    data_path=DATA_PATH,
                                    step_num=step_num,
                                    sn_per_worker_gb=_sn_per_worker,
                                    use_live_vram_poll=_use_live_poll,
                                    gpu_headroom_gb=_gpu_headroom,
                                )
    
                            # Generate HTML sensitivity plot — mirrors what AIMET's
                            # perform_per_layer_analysis_by_enabling_quantizers writes
                            # in the sequential path ({step_mmp_dir}/per_layer_quant_enabled.html).
                            try:
                                export_per_layer_sensitivity_analysis_plot(
                                    sensitivity_data,
                                    os.path.join(step_mmp_dir, "per_layer_quant_enabled"),
                                    title="per_layer_quant_enabled",
                                )
                                qs_logger.debug(
                                    "MMP step-%d: exported sensitivity HTML plot → %s",
                                    step_num,
                                    os.path.join(step_mmp_dir, "per_layer_quant_enabled.html"),
                                )
                            except Exception as _plot_exc:
                                qs_logger.warning(
                                    "MMP step-%d: failed to export sensitivity HTML plot: %s",
                                    step_num, _plot_exc,
                                )
                        else:
                            sensitivity_data = quanter.run_per_layer_sensitivity(
                                sim=sim,
                                dummy_input=dummy_input,
                                full_forward=full_forward,
                                full_eval=full_eval,
                                mmp_results_dir=step_mmp_dir,
                            )
                        with open(
                            os.path.join(
                                qscheme_results_dir,
                                f"per_layer_sensitivity_step{step_num}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(sensitivity_data, f, indent=4)
    
                        # Compute threshold from the mean of this step's sensitivity scores.
                        # Only include active-op scores that are finite (exclude nan from
                        # workers that failed even after CPU fallback).
                        _active_scores = [
                            score for name, score in sensitivity_data.items()
                            if name in active_ops and np.isfinite(score)
                        ]
                        sensitivity_mean_metric = (
                            float(np.mean(_active_scores)) if _active_scores
                            else reference_metrics["sqnr_db"]
                        )
                        sensitivity_threshold = sensitivity_mean_metric - MMP_SENSITIVITY_MARGIN_DB
                        qs_logger.debug(
                            "MMP step-%d sensitivity threshold: "
                            "mean=%.2f dB  margin=%.1f dB  threshold=%.2f dB  "
                            "(n_scores=%d)",
                            step_num, sensitivity_mean_metric, MMP_SENSITIVITY_MARGIN_DB,
                            sensitivity_threshold, len(_active_scores),
                        )
    
                        # Find sensitive layers at candidate_precision.
                        sensitive_layers = [
                            (name, score)
                            for name, score in sensitivity_data.items()
                            if name in active_ops
                            and layer_precision.get(name) == candidate_precision
                            and score < sensitivity_threshold
                        ]
    
                        if not sensitive_layers:
                            qs_logger.debug(
                                "MMP step-%d: no sensitive layers at '%s' below "
                                "threshold=%.2f dB — done.",
                                step_num, candidate_precision, sensitivity_threshold,
                            )
                            break
    
                        n_flipped = 0
                        if is_two_tier:
                            tier_low, tier_high = step
                            tier_low_label  = _fallback_label(tier_low)
                            tier_high_label = _fallback_label(tier_high)
                            n_low = n_high = 0
                            for layer_name, score in sensitive_layers:
                                if score < sensitivity_threshold - MMP_TIER_SPLIT_EPSILON_DB:
                                    # Very sensitive → high tier (e.g. fp16)
                                    n_flipped += _apply_step_to_layer(layer_name, tier_high)
                                    layer_precision[layer_name] = tier_high_label
                                    n_high += 1
                                    qs_logger.debug(
                                        "  flip %-40s  SQNR=%.2f dB  %s → %s  [high tier]",
                                        layer_name, score, candidate_precision, tier_high_label,
                                    )
                                else:
                                    # Moderately sensitive → low tier (e.g. w8a16)
                                    n_flipped += _apply_step_to_layer(layer_name, tier_low)
                                    layer_precision[layer_name] = tier_low_label
                                    n_low += 1
                                    qs_logger.debug(
                                        "  flip %-40s  SQNR=%.2f dB  %s → %s  [low tier]",
                                        layer_name, score, candidate_precision, tier_low_label,
                                    )
                            qs_logger.debug(
                                "MMP step-%d (two-tier): flipped %d layers (%d quantizers): "
                                "%d → %s, %d → %s.",
                                step_num, len(sensitive_layers), n_flipped,
                                n_low, tier_low_label, n_high, tier_high_label,
                            )
                        else:
                            for layer_name, score in sensitive_layers:
                                n_flipped += _apply_step_to_layer(layer_name, step)
                                layer_precision[layer_name] = step_label
                                qs_logger.debug(
                                    "  flip %-40s  SQNR=%.2f dB  %s → %s",
                                    layer_name, score, candidate_precision, step_label,
                                )
                            qs_logger.debug(
                                "MMP step-%d: flipped %d layers (%d quantizers) to '%s'.",
                                step_num, len(sensitive_layers), n_flipped, step_label,
                            )
    
                        _ensure_qdq_mode()
    
                        # Re-compute encodings after precision flips.
                        qs_logger.debug("MMP step-%d: re-computing encodings ...", step_num)
                        sim.compute_encodings(quanter.forward_pass_callback)
    
                        # Evaluate the mixed model.
                        step_metrics = quanter.eval_sim(sim, label=f"MMP-step{step_num}-{step_label}")
                        step_sqnr  = step_metrics["sqnr_db"]
                        step_delta = abs(step_sqnr - reference_metrics["sqnr_db"])
                        qs_logger.debug(
                            "MMP step-%d SQNR: %.2f dB  delta=%.2f dB  fp16_ref=%.2f dB  %s",
                            step_num, step_sqnr, step_delta, reference_metrics["sqnr_db"],
                            "within threshold" if step_delta < MMP_SKIP_SQNR_DELTA_DB
                            else "still degraded",
                        )
    
                        if step_delta < MMP_SKIP_SQNR_DELTA_DB:
                            qs_logger.debug("MMP converged at step-%d.", step_num)
                            step_converged = True
                            break
    
                    # Build final per-layer outcome lists.
                    mmp_flipped_layers: list[tuple[str, float, str]] = [
                        (name, sensitivity_data.get(name, float("nan")), prec)
                        for name, prec in layer_precision.items()
                        if prec != qscheme_key
                    ]
                    mmp_remaining_layers: list[tuple[str, float]] = [
                        (name, sensitivity_data.get(name, float("nan")))
                        for name, prec in layer_precision.items()
                        if prec == qscheme_key
                    ]
    
                    n_enabled_after = sum(1 for q in sim.qc_quantize_op_dict.values() if q.enabled)
                    available       = pre_flip_enabled   # = total - htp_disabled
    
                    # Quantizer breakdown table (dynamic per-precision counts)
                    from collections import Counter
                    prec_counts = Counter(layer_precision.values())
    
                    _c1, _c2, _c3 = 32, 9, 42
                    _sep  = f"├{'─'*(_c1+2)}┼{'─'*(_c2+2)}┼{'─'*(_c3+2)}┤"
                    _top  = f"┌{'─'*(_c1+2)}┬{'─'*(_c2+2)}┬{'─'*(_c3+2)}┐"
                    _bot  = f"└{'─'*(_c1+2)}┴{'─'*(_c2+2)}┴{'─'*(_c3+2)}┘"
                    def _qrow(label, count, note=""):
                        return f"│ {label:<{_c1}} │ {str(count):>{_c2}} │ {note:<{_c3}} │"
    
                    _prec_note = "  ".join(f"{p}={c}" for p, c in sorted(prec_counts.items()))
                    _qtable_rows = [
                        _top,
                        _qrow("Category",             "Count",  "Notes"),
                        _sep,
                        _qrow("Total quantizers",      total_q,       ""),
                        _qrow("  disabled by htp_v81", n_htp_disabled, "Config file — never touched by MMP"),
                        _sep,
                        _qrow("Available to MMP",      available,     ""),
                    ]
                    for prec, cnt in sorted(prec_counts.items()):
                        if prec == qscheme_key:
                            note = "remaining at base precision"
                        else:
                            note = "promoted by MMP"
                        _qtable_rows.append(_qrow(f"  layers at {prec}", cnt, note))
                    _qtable_rows.extend([
                        _sep,
                        _qrow("Layer precision split", "",  _prec_note),
                        _bot,
                    ])
                    qs_logger.debug("MMP quantizer breakdown:\n%s", "\n".join(_qtable_rows))
    
                    # Per-layer breakdown table — logged at DEBUG so it lands in run.log
                    print_mmp_layer_table(
                        qscheme_key=qscheme_key,
                        sensitivity_data=sensitivity_data,
                        htp_disabled_layers=htp_disabled_layers,
                        flipped_layers=mmp_flipped_layers,
                        remaining_layers=mmp_remaining_layers,
                        logger=qs_logger,
                    )
    
                    mmp_prefix = f"{module_name}_{qscheme_key}_mmp"
                    sim.export(path=mmp_results_dir, filename_prefix=mmp_prefix, export_model=True)
                    mmp_metrics = quanter.eval_sim(sim, label="MMP")
    
            # Explicitly delete sim so AIMET's OrtInferenceSession.__del__ runs now,
            # while the logging module is still alive, suppressing the shutdown-time
            # "AttributeError: 'NoneType' object has no attribute 'warning'" noise.
            del sim
    
            table_rows.append({
                "qscheme":       qscheme_key,
                "quant_metrics": quant_metrics,
                "mmp_metrics":   mmp_metrics,
            })
            all_results[qscheme_key] = {
                "reference": reference_metrics,   # fp16 anchor
                "quant":     quant_metrics,
                "mmp":       mmp_metrics,
            }
    
        # ------------------------------------------------------------------
        # Print summary table and save JSON — once per module
        # ------------------------------------------------------------------
        quanter.logger = logger   # restore module-level logger
        display_name = (
            f"{module_name} [{opt_tag}]"
            if module_alias == module_name
            else f"{module_name} → {module_alias} [{opt_tag}]"
        )
        # orig_key = "fp16"; reference_metrics = fp16 quant metrics
        # fp32_info_metrics shown as informational row (not on device)
        print_results_table(
            display_name,
            orig_key,
            reference_metrics,
            table_rows,
            logger,
            fp32_info_metrics=fp32_metrics,
        )
    
        with open(os.path.join(results_dir, "final_eval_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
