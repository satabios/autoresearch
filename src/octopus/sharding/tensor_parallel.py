from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from octopus._logging import get_logger
from octopus.adapters.base import ModelAdapter
from octopus.exceptions import ShardingError
from octopus.sharding.base import ShardingStrategy

_log = get_logger()


@dataclass
class _ShardedLinear:
    """One Linear layer sharded across output features on multiple GPUs."""

    devices: list[torch.device]
    weights: list[torch.Tensor]
    biases: list[torch.Tensor | None]
    primary_device: torch.device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for device, weight, bias in zip(self.devices, self.weights, self.biases):
            x_dev = x.to(device, non_blocking=True)
            y_dev = F.linear(x_dev, weight, bias)
            parts.append(y_dev.to(self.primary_device, non_blocking=True))
        return torch.cat(parts, dim=-1)


class _TPModelWrapper:
    """Callable wrapper passed into eval_fn for tensor-parallel inference."""

    def __init__(
        self,
        ops: list[Any],
        eval_fn: Any,
        primary_device: torch.device,
    ) -> None:
        self._ops = ops
        self.eval_fn = eval_fn
        self.primary_device = primary_device

    def __call__(self, batch: Any) -> Any:
        with torch.no_grad():
            x = _to_device(batch, self.primary_device)
            for op in self._ops:
                x = op(x)
            return x


def _split_sizes(total: int, parts: int) -> list[int]:
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    return batch


class TensorParallelStrategy(ShardingStrategy):
    """Tensor parallelism for PyTorch sequential linear stacks.

    Current runtime scope:
    - single process with multiple visible CUDA devices
    - `nn.Sequential` models made of `nn.Linear` + elementwise ops
    - each Linear split by output features across GPUs, then concatenated
    """

    def compute_gpus_needed(self, model_vram_gb: float, gpu_vram_gb: float) -> int:
        # Linear weight sharding scales almost linearly; keep conservative overhead.
        overhead = 1.10
        return math.ceil(model_vram_gb * overhead / gpu_vram_gb)

    def shard_model(self, adapter: ModelAdapter, device_ids: list[int]) -> Any:
        if len(device_ids) < 2:
            raise ShardingError("Tensor parallel requires at least 2 GPUs.")

        # Build from CPU weights to avoid duplicating full model on a CUDA device first.
        adapter.load_to_device(torch.device("cpu"))
        model = adapter._active_model  # type: ignore[attr-defined]
        if model is None:
            raise ShardingError("Adapter model not loaded after load_to_device().")
        if not isinstance(model, nn.Sequential):
            raise ShardingError(
                "Current TP runtime supports nn.Sequential models only. "
                "Use sharding_strategy='pp' for other PyTorch model topologies."
            )

        devices = [torch.device(f"cuda:{d}") for d in device_ids]
        primary = devices[0]
        ops: list[Any] = []

        for idx, module in enumerate(model):
            if isinstance(module, nn.Linear):
                shard_sizes = _split_sizes(module.out_features, len(devices))
                weight_slices = torch.split(module.weight.detach().cpu(), shard_sizes, dim=0)
                if module.bias is None:
                    bias_slices = [None] * len(devices)
                else:
                    bias_slices = list(torch.split(module.bias.detach().cpu(), shard_sizes, dim=0))

                shard = _ShardedLinear(
                    devices=devices,
                    weights=[w.to(dev, non_blocking=True) for w, dev in zip(weight_slices, devices)],
                    biases=[
                        (b.to(dev, non_blocking=True) if b is not None else None)
                        for b, dev in zip(bias_slices, devices)
                    ],
                    primary_device=primary,
                )
                ops.append(shard)
                continue

            # Keep non-linear / elementwise modules on primary GPU.
            op_module = copy.deepcopy(module).to(primary)
            op_module.eval()
            ops.append(op_module)

        _log.info(
            "Applied TP: %d ops over %d GPUs for Sequential model.",
            len(ops),
            len(devices),
        )
        return _TPModelWrapper(ops=ops, eval_fn=adapter.eval_fn, primary_device=primary)

    def run_sharded(self, sharded_model: Any, batch: Any) -> Any:
        wrapper: _TPModelWrapper = sharded_model
        return wrapper.eval_fn(wrapper, batch)
