from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import ray
import torch

from octopus._logging import get_logger
from octopus.adapters import resolve_adapter_class
from octopus.exceptions import WorkerOOMError

_log = get_logger()


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

    Coordinates a torch.distributed process group internally for TP/PP.
    """

    def __init__(
        self,
        model_bytes: bytes,
        adapter_cls_name: str,
        eval_fn: Callable,
        device_ids: list[int],
        sharding_strategy: Literal["tp", "pp"],
        group_rank: int,
    ) -> None:
        self._model_bytes = model_bytes
        self._adapter_cls_name = adapter_cls_name
        self._eval_fn = eval_fn
        self._device_ids = device_ids
        self._strategy_name = sharding_strategy
        self._group_rank = group_rank
        self._sharding_strategy: Optional[Any] = None
        self._sharded_model: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> dict:
        """Set up the sharded model across devices."""
        from octopus.sharding import get_strategy

        adapter_cls = resolve_adapter_class(self._adapter_cls_name)
        adapter = adapter_cls.from_state_bytes(self._model_bytes, self._eval_fn)

        self._sharding_strategy = get_strategy(self._strategy_name)
        self._sharded_model = self._sharding_strategy.shard_model(
            adapter, self._device_ids
        )
        self._initialized = True
        return {
            "status": "ready",
            "device_ids": self._device_ids,
            "strategy": self._strategy_name,
        }

    def run(self, batch: Any) -> Any:
        """Run sharded inference."""
        if not self._initialized:
            raise RuntimeError("Sharded worker not initialized.")
        try:
            return self._sharding_strategy.run_sharded(self._sharded_model, batch)
        except torch.cuda.OutOfMemoryError as e:
            raise WorkerOOMError(f"OOM during sharded inference: {e}") from e

    def shutdown(self) -> None:
        """Release resources."""
        self._sharded_model = None
        self._sharding_strategy = None
        self._initialized = False
        torch.cuda.empty_cache()
