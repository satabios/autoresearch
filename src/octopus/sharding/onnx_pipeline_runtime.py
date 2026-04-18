from __future__ import annotations

import logging
import tempfile
from typing import Any, Optional

import numpy as np
import torch

from octopus.adapters.onnx_rt import extract_model_bundle_to_tempdir
from octopus.exceptions import ShardingError
from octopus.sharding.onnx_partition import materialize_stage_models, plan_stages_from_onnx

_log = logging.getLogger(__name__)


class ORTPipelineSessionProxy:
    """Minimal InferenceSession-like proxy for staged ONNX pipeline."""

    def __init__(
        self,
        stage_sessions: list[Any],
        stage_input_names: list[tuple[str, ...]],
        stage_output_names: list[tuple[str, ...]],
        temp_dirs: list[tempfile.TemporaryDirectory[str]],
    ) -> None:
        self._stage_sessions = stage_sessions
        self._stage_input_names = stage_input_names
        self._stage_output_names = stage_output_names
        self._temp_dirs = temp_dirs

    def run(
        self,
        output_names: Optional[list[str]],
        input_feed: Any,
        *_,
        **__,
    ) -> list[np.ndarray]:
        """Execute staged pipeline with ORT-compatible run signature."""
        current_feed = normalize_ort_input_feed(input_feed, self._stage_input_names[0])
        final_outputs: dict[str, np.ndarray] = {}

        for stage_idx, session in enumerate(self._stage_sessions):
            stage_outputs = list(self._stage_output_names[stage_idx])
            stage_values = session.run(stage_outputs, current_feed)
            produced = {name: value for name, value in zip(stage_outputs, stage_values)}

            if stage_idx == len(self._stage_sessions) - 1:
                final_outputs = produced
                break

            next_inputs = self._stage_input_names[stage_idx + 1]
            next_feed: dict[str, Any] = {}
            for name in next_inputs:
                if name in produced:
                    next_feed[name] = produced[name]
                elif name in current_feed:
                    # Pass-through tensors (e.g., masks) used by later stages.
                    next_feed[name] = current_feed[name]
                else:
                    raise ShardingError(
                        f"Missing boundary tensor {name!r} from stage {stage_idx} "
                        f"to stage {stage_idx + 1}."
                    )
            current_feed = next_feed

        if output_names is None:
            output_names = list(self._stage_output_names[-1])
        return [final_outputs[name] for name in output_names]

    def close(self) -> None:
        for d in self._temp_dirs:
            try:
                d.cleanup()
            except OSError as e:
                _log.debug("Temp dir cleanup failed: %s", e)
        self._temp_dirs.clear()
        self._stage_sessions.clear()


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def normalize_ort_input_feed(input_feed: Any, input_names: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(input_feed, dict):
        return {str(k): _to_numpy(v) for k, v in input_feed.items()}
    if len(input_names) != 1:
        raise ShardingError(
            "Non-dict batch provided, but staged ONNX model expects multiple inputs."
        )
    return {input_names[0]: _to_numpy(input_feed)}


def build_onnx_pipeline_proxy(
    model_bytes: bytes,
    local_device_ids: list[int],
) -> ORTPipelineSessionProxy:
    """Materialize staged ONNX models and create per-stage ORT sessions."""
    if len(local_device_ids) < 2:
        raise ShardingError("ONNX pipeline parallel requires at least 2 GPUs per shard group.")
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as e:
        raise ShardingError(
            "onnxruntime is required for ONNX shared-model pipeline runtime."
        ) from e

    model_tmpdir, model_path = extract_model_bundle_to_tempdir(model_bytes)
    stage_tmpdir = tempfile.TemporaryDirectory()
    try:
        stage_plans = plan_stages_from_onnx(model_path, num_stages=len(local_device_ids))
        artifacts = materialize_stage_models(
            model_path,
            stage_plans,
            output_dir=stage_tmpdir.name,
        )
        stage_sessions: list[Any] = []
        stage_input_names: list[tuple[str, ...]] = []
        stage_output_names: list[tuple[str, ...]] = []
        for artifact, local_gpu_id in zip(artifacts, local_device_ids):
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"device_id": local_gpu_id, "use_tf32": 0},
                ),
                "CPUExecutionProvider",
            ]
            session = ort.InferenceSession(artifact.onnx_path, providers=providers)
            stage_sessions.append(session)
            stage_input_names.append(artifact.input_names)
            stage_output_names.append(artifact.output_names)
        return ORTPipelineSessionProxy(
            stage_sessions=stage_sessions,
            stage_input_names=stage_input_names,
            stage_output_names=stage_output_names,
            temp_dirs=[model_tmpdir, stage_tmpdir],
        )
    except Exception:
        model_tmpdir.cleanup()
        stage_tmpdir.cleanup()
        raise
