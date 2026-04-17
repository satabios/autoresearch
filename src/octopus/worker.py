from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import ray
import torch

from octopus._logging import get_logger
from octopus.adapters import resolve_adapter_class
from octopus.exceptions import ShardingError, WorkerOOMError, detect_oom_in_logs
from octopus.sharding.onnx_pipeline_runtime import (
    ORTPipelineSessionProxy as _ORTPipelineSessionProxy,
)
from octopus.sharding.onnx_pipeline_runtime import (
    build_onnx_pipeline_proxy as _build_onnx_pipeline_proxy,
)
from octopus.sharding.onnx_pipeline_runtime import (
    normalize_ort_input_feed as _normalize_ort_input_feed,
)

_log = get_logger()


def _build_op_name_to_quantizers(sim: Any) -> dict[str, list[Any]]:
    """Map op name → list of quantizers for that op.

    Walks sim.qc_quantize_op_dict (aimet_onnx) to build the mapping.
    Falls back to empty dict if the attribute doesn't exist.
    """
    mapping: dict[str, list[Any]] = {}
    qc_dict = getattr(sim, "qc_quantize_op_dict", None)
    if qc_dict is None:
        return mapping
    for op_name, quantizers in qc_dict.items():
        if isinstance(quantizers, (list, tuple)):
            mapping[op_name] = list(quantizers)
        else:
            mapping[op_name] = [quantizers]
    return mapping


def _snapshot_quantizers(op_name_to_quantizers: dict[str, list[Any]]) -> dict[int, tuple]:
    """Snapshot enabled/dtype/bitwidth/op_mode for every quantizer.

    Returns {id(q): (enabled, data_type, bitwidth, op_mode)}.
    """
    snapshot: dict[int, tuple] = {}
    for quantizers in op_name_to_quantizers.values():
        for q in quantizers:
            snapshot[id(q)] = (
                q.enabled,
                getattr(q, "data_type", None),
                getattr(q, "bitwidth", None),
                getattr(q, "op_mode", None),
            )
    return snapshot


def _disable_all_quantizers(op_name_to_quantizers: dict[str, list[Any]]) -> None:
    """Disable every quantizer in the map."""
    for quantizers in op_name_to_quantizers.values():
        for q in quantizers:
            q.enabled = False


def _restore_quantizers(
    op_name_to_quantizers: dict[str, list[Any]],
    snapshot: dict[int, tuple],
) -> None:
    """Restore quantizers to their snapshotted state."""
    for quantizers in op_name_to_quantizers.values():
        for q in quantizers:
            state = snapshot.get(id(q))
            if state is None:
                continue
            enabled, data_type, bitwidth, op_mode = state
            q.enabled = enabled
            if data_type is not None:
                q.data_type = data_type
            if bitwidth is not None:
                q.bitwidth = bitwidth
            if op_mode is not None:
                q.op_mode = op_mode


def _run_enabling_loop(
    sim: Any,
    layer_names: list[str],
    op_name_to_quantizers: dict[str, list[Any]],
    eval_fn: Callable,
) -> dict[str, float]:
    """Core enabling loop: snapshot → disable all → enable one → eval → disable → restore.

    Args:
        sim: aimet_onnx QuantizationSimModel (already loaded).
        layer_names: op names to evaluate.
        op_name_to_quantizers: pre-built mapping from _build_op_name_to_quantizers().
        eval_fn: (ort.InferenceSession) -> float.

    Returns:
        {layer_name: sqnr_score}
    """
    try:
        try:
            from aimet_onnx.qc_quantize_op import OpMode  # type: ignore[import-untyped]
        except ImportError:
            from aimet_common.defs import OpMode  # type: ignore[import-untyped]
        op_mode_qdq = OpMode.quantizeDequantize
    except ImportError:
        op_mode_qdq = None

    snapshot = _snapshot_quantizers(op_name_to_quantizers)
    _disable_all_quantizers(op_name_to_quantizers)

    results: dict[str, float] = {}
    for layer_name in layer_names:
        layer_quantizers = op_name_to_quantizers.get(layer_name, [])

        # Enable this layer's quantizers from snapshot
        for q in layer_quantizers:
            state = snapshot.get(id(q))
            if state is not None:
                enabled, data_type, bitwidth, op_mode = state
                q.enabled = enabled
                if data_type is not None:
                    q.data_type = data_type
                if bitwidth is not None:
                    q.bitwidth = bitwidth
            if op_mode_qdq is not None:
                q.op_mode = op_mode_qdq

        try:
            score = eval_fn(sim.session)
        except Exception as e:
            _log.warning("eval_fn failed for layer %s: %s", layer_name, e)
            score = float("nan")

        results[layer_name] = score

        # Disable this layer's quantizers before moving to next
        for q in layer_quantizers:
            q.enabled = False

    _restore_quantizers(op_name_to_quantizers, snapshot)
    return results


def _current_actor_gpu_ids() -> list[int]:
    """Return physical GPU ids Ray assigned to current actor."""
    try:
        gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    except Exception:
        return []
    result: list[int] = []
    for gpu_id in gpu_ids:
        try:
            result.append(int(gpu_id))
        except (TypeError, ValueError):
            continue
    return result


@ray.remote
class InferenceWorker:
    """Ray actor that owns one model copy on one GPU.

    Lifecycle:
        1. Constructed with serialized model bytes + eval_fn
        2. initialize() loads model onto GPU (cuda:0 within its CUDA_VISIBLE_DEVICES scope)
        3. run(batch) runs inference
        4. shutdown() frees GPU memory
    """

    def __init__(
        self,
        model_bytes: bytes,
        adapter_cls_name: str,
        eval_fn: Callable,
    ) -> None:
        self._model_bytes = model_bytes
        self._adapter_cls_name = adapter_cls_name
        self._eval_fn = eval_fn
        self._adapter: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> dict:
        """Load model onto GPU. Returns status dict."""
        adapter_cls = resolve_adapter_class(self._adapter_cls_name)
        self._adapter = adapter_cls.from_state_bytes(self._model_bytes, self._eval_fn)
        # Always cuda:0 because CUDA_VISIBLE_DEVICES is set per-actor
        device = torch.device("cuda:0")
        self._adapter.load_to_device(device)
        self._initialized = True
        mem_mb = torch.cuda.memory_allocated(0) / (1 << 20)
        return {
            "status": "ready",
            "memory_allocated_mb": mem_mb,
        }

    def run(self, batch: Any) -> Any:
        """Run inference on a single batch."""
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call initialize() first.")
        try:
            return self._adapter.forward(batch)
        except torch.cuda.OutOfMemoryError as e:
            raise WorkerOOMError(f"OOM during inference: {e}") from e
        except RuntimeError as e:
            pattern = detect_oom_in_logs(str(e))
            if pattern:
                raise WorkerOOMError(
                    f"OOM during inference: {e}",
                    oom_pattern=pattern,
                ) from e
            raise

    def health_check(self) -> bool:
        """Returns True if worker is alive and GPU is accessible."""
        try:
            torch.cuda.memory_allocated(0)
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        """Release GPU resources."""
        if self._adapter is not None:
            self._adapter.unload()
            self._adapter = None
        self._initialized = False


@ray.remote
class ShardedInferenceWorkerGroup:
    """A logical worker that spans multiple GPUs for one sharded model.

    Validated runtime today: PyTorch + pipeline parallel.
    """

    def __init__(
        self,
        model_bytes: bytes,
        adapter_cls_name: str,
        eval_fn: Callable,
        planned_device_ids: list[int],
        sharding_strategy: Literal["tp", "pp"],
        group_rank: int,
    ) -> None:
        self._model_bytes = model_bytes
        self._adapter_cls_name = adapter_cls_name
        self._eval_fn = eval_fn
        self._planned_device_ids = planned_device_ids
        self._strategy_name = sharding_strategy
        self._group_rank = group_rank
        self._adapter: Optional[Any] = None
        self._sharding_strategy: Optional[Any] = None
        self._sharded_model: Optional[Any] = None
        self._assigned_gpu_ids: list[int] = []
        self._local_device_ids: list[int] = []
        self._initialized = False

    def initialize(self) -> dict:
        """Set up the sharded model across devices."""
        from octopus.sharding import get_strategy

        if self._adapter_cls_name not in {"pytorch", "onnx", "onnx_quantsim"}:
            raise ShardingError(
                "Shared-model multi-GPU runtime currently supports "
                "PyTorch/ORT/AIMET ONNX only."
            )
        if self._strategy_name == "tp" and self._adapter_cls_name != "pytorch":
            raise ShardingError(
                "sharding_strategy='tp' currently supported for PyTorch only."
            )

        self._assigned_gpu_ids = _current_actor_gpu_ids()
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count <= 0:
            raise ShardingError("Sharded worker started without visible CUDA devices.")
        self._local_device_ids = list(range(visible_gpu_count))
        if (
            self._planned_device_ids
            and visible_gpu_count != len(self._planned_device_ids)
        ):
            _log.warning(
                "Sharded worker %d planned for %s but sees %d GPU(s): assigned=%s",
                self._group_rank,
                self._planned_device_ids,
                visible_gpu_count,
                self._assigned_gpu_ids,
            )

        if self._adapter_cls_name in {"onnx", "onnx_quantsim"}:
            if self._strategy_name != "pp":
                raise ShardingError(
                    "ONNX-family shared-model runtime supports sharding_strategy='pp' only."
                )
            self._sharding_strategy = None
            self._sharded_model = _build_onnx_pipeline_proxy(
                self._model_bytes,
                self._local_device_ids,
            )
        else:
            adapter_cls = resolve_adapter_class(self._adapter_cls_name)
            self._adapter = adapter_cls.from_state_bytes(self._model_bytes, self._eval_fn)

            self._sharding_strategy = get_strategy(self._strategy_name)
            self._sharded_model = self._sharding_strategy.shard_model(
                self._adapter, self._local_device_ids
            )
        self._initialized = True
        return {
            "status": "ready",
            "planned_device_ids": self._planned_device_ids,
            "assigned_gpu_ids": self._assigned_gpu_ids,
            "visible_gpu_count": visible_gpu_count,
            "strategy": self._strategy_name,
            "adapter": self._adapter_cls_name,
        }

    def run(self, batch: Any) -> Any:
        """Run sharded inference."""
        if not self._initialized:
            raise RuntimeError("Sharded worker not initialized.")
        try:
            if self._adapter_cls_name == "onnx":
                return self._eval_fn(self._sharded_model, batch)
            if self._adapter_cls_name == "onnx_quantsim":
                return self._eval_fn(self._sharded_model)
            return self._sharding_strategy.run_sharded(self._sharded_model, batch)
        except torch.cuda.OutOfMemoryError as e:
            raise WorkerOOMError(f"OOM during sharded inference: {e}") from e

    def shutdown(self) -> None:
        """Release resources."""
        if self._adapter is not None:
            self._adapter.unload()
            self._adapter = None
        if isinstance(self._sharded_model, _ORTPipelineSessionProxy):
            self._sharded_model.close()
        self._sharded_model = None
        self._sharding_strategy = None
        self._initialized = False
        torch.cuda.empty_cache()


@ray.remote(num_gpus=0)
class SensitivityWorker:
    """Ray actor for enabling-loop sensitivity analysis.

    Unlike InferenceWorker (one batch per call), this worker:
    - Loads the full QuantSim model once at initialize()
    - Receives a list of layer names via process_layers()
    - Runs the enabling loop (snapshot → disable all → enable one → eval → disable)
    - Returns {layer_name: sqnr_score}
    - Accepts additional layers mid-run via steal_layers()

    Lifecycle:
        1. Constructed with serialized model bytes + eval_fn
        2. initialize() reconstructs adapter, builds op→quantizer map
        3. process_layers(layer_names) runs enabling loop, accumulates results
        4. steal_layers(extra) appends extra layers and processes them
        5. get_results() returns accumulated {layer: score} dict
        6. shutdown() frees resources
    """

    def __init__(
        self,
        model_bytes: bytes,
        adapter_cls_name: str,
        eval_fn: Callable,
    ) -> None:
        self._model_bytes = model_bytes
        self._adapter_cls_name = adapter_cls_name
        self._eval_fn = eval_fn
        self._adapter: Optional[Any] = None
        self._quantsim: Optional[Any] = None
        self._op_name_to_quantizers: dict[str, list[Any]] = {}
        self._results: dict[str, float] = {}
        self._initialized = False

    def initialize(self) -> dict:
        """Reconstruct adapter and build op→quantizer mapping. Returns status dict."""
        try:
            adapter_cls = resolve_adapter_class(self._adapter_cls_name)
            self._adapter = adapter_cls.from_state_bytes(self._model_bytes, self._eval_fn)
        except Exception as e:
            pattern = detect_oom_in_logs(str(e))
            if pattern:
                raise WorkerOOMError(
                    f"GPU session init failed: {e}",
                    oom_pattern=pattern,
                ) from e
            raise
        sim = getattr(self._adapter, "sim", getattr(self._adapter, "_sim", None))
        self._op_name_to_quantizers = _build_op_name_to_quantizers(sim)
        self._quantsim = sim  # cached reference for the enabling loop
        self._initialized = True
        return {
            "status": "ready",
            "num_quantizer_ops": len(self._op_name_to_quantizers),
        }

    def process_layers(self, layer_names: list[str]) -> dict[str, float]:
        """Run enabling loop for the given layers. Accumulates into get_results().

        Args:
            layer_names: op names to evaluate.

        Returns:
            {layer_name: sqnr_score} for this batch.
        """
        if not self._initialized:
            raise RuntimeError("SensitivityWorker not initialized. Call initialize() first.")
        batch_results = _run_enabling_loop(
            self._quantsim,
            layer_names,
            self._op_name_to_quantizers,
            self._eval_fn,
        )
        self._results.update(batch_results)
        return batch_results

    def steal_layers(self, extra: list[str]) -> dict[str, float]:
        """Accept additional layers stolen from the dynamic queue and process them.

        Called by DynamicScheduler when a GPU has spare VRAM headroom.

        Args:
            extra: additional layer names to evaluate.

        Returns:
            {layer_name: sqnr_score} for the stolen layers.
        """
        return self.process_layers(extra)

    def get_status(self) -> dict:
        """Return current worker status."""
        return {
            "initialized": self._initialized,
            "results_so_far": len(self._results),
        }

    def get_results(self) -> dict[str, float]:
        """Return all accumulated {layer_name: sqnr_score} results."""
        return dict(self._results)

    def shutdown(self) -> None:
        """Release GPU resources."""
        if self._adapter is not None:
            self._adapter.unload()
            self._adapter = None
        self._quantsim = None
        self._op_name_to_quantizers = {}
        self._initialized = False
