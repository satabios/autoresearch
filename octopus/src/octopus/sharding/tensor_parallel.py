from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from octopus._logging import get_logger
from octopus.adapters.base import ModelAdapter
from octopus.exceptions import ShardingError
from octopus.sharding.base import ShardingStrategy

_log = get_logger()


class _TPModelWrapper:
    """Wraps a tensor-parallelized model for inference."""

    def __init__(
        self,
        model: nn.Module,
        device_mesh: Any,
        eval_fn: Any,
        primary_device: torch.device,
    ) -> None:
        self.model = model
        self.device_mesh = device_mesh
        self.eval_fn = eval_fn
        self.primary_device = primary_device


class TensorParallelStrategy(ShardingStrategy):
    """Tensor Parallelism: split individual layers across GPUs.

    Uses torch.distributed.tensor.parallel (PyTorch 2.x TP API).
    Each GPU holds a slice of every weight matrix. All GPUs process
    the same batch simultaneously, communicating via NCCL all-reduce.
    """

    def compute_gpus_needed(self, model_vram_gb: float, gpu_vram_gb: float) -> int:
        # TP scales roughly linearly with 10% comm overhead
        overhead = 1.10
        return math.ceil(model_vram_gb * overhead / gpu_vram_gb)

    def shard_model(self, adapter: ModelAdapter, device_ids: list[int]) -> Any:
        """Apply tensor parallelism across device_ids.

        Steps:
            1. Init NCCL process group
            2. Create DeviceMesh
            3. Apply ColwiseParallel/RowwiseParallel to Linear layers
            4. Return wrapped model
        """
        try:
            from torch.distributed.tensor import DeviceMesh
            from torch.distributed.tensor.parallel import (
                ColwiseParallel,
                RowwiseParallel,
                parallelize_module,
            )
        except ImportError as e:
            raise ShardingError(
                "Tensor Parallel requires PyTorch >= 2.1 with torch.distributed.tensor. "
                f"Import error: {e}"
            ) from e

        # Initialize process group if not already
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        device_mesh = DeviceMesh("cuda", device_ids)

        # Load model to first device initially
        primary_device = torch.device(f"cuda:{device_ids[0]}")
        adapter.load_to_device(primary_device)

        # Access the underlying model
        model = adapter._active_model  # type: ignore[attr-defined]
        if model is None:
            raise ShardingError("Adapter model not loaded after load_to_device().")

        # Auto-detect and parallelize Linear layers
        plan = self._build_parallel_plan(model)
        if plan:
            parallelize_module(model, device_mesh, plan)
            _log.info("Applied TP to %d layer groups across %d GPUs.", len(plan), len(device_ids))
        else:
            _log.warning("No parallelizable layers found — model will be replicated.")

        model.eval()
        return _TPModelWrapper(model, device_mesh, adapter.eval_fn, primary_device)

    def run_sharded(self, sharded_model: Any, batch: Any) -> Any:
        wrapper: _TPModelWrapper = sharded_model
        with torch.no_grad():
            if isinstance(batch, torch.Tensor):
                batch = batch.to(wrapper.primary_device, non_blocking=True)
            elif isinstance(batch, dict):
                batch = {
                    k: v.to(wrapper.primary_device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            return wrapper.eval_fn(wrapper.model, batch)

    @staticmethod
    def _build_parallel_plan(model: nn.Module) -> dict:
        """Build a TP plan by inspecting the model structure.

        Heuristic: for transformer-like models, applies ColwiseParallel to
        projection-up layers (q, k, v, fc1/up) and RowwiseParallel to
        projection-down layers (out_proj, fc2/down).
        """
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        colwise_patterns = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "c_fc", "c_attn"}
        rowwise_patterns = {"out_proj", "o_proj", "down_proj", "c_proj"}

        plan: dict[str, Any] = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            short_name = name.split(".")[-1]
            if short_name in colwise_patterns:
                plan[name] = ColwiseParallel()
            elif short_name in rowwise_patterns:
                plan[name] = RowwiseParallel()

        return plan
