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


class _PPStage:
    """One pipeline stage: a subset of layers on a specific device."""

    def __init__(self, layers: nn.Sequential, device: torch.device) -> None:
        self.layers = layers
        self.device = device


class _PPModelWrapper:
    """Wraps a pipeline-parallelized model for inference."""

    def __init__(
        self,
        stages: list[_PPStage],
        embedding: nn.Module | None,
        head: nn.Module | None,
        eval_fn: Any,
        first_device: torch.device,
        last_device: torch.device,
    ) -> None:
        self.stages = stages
        self.embedding = embedding
        self.head = head
        self.eval_fn = eval_fn
        self.first_device = first_device
        self.last_device = last_device


class PipelineParallelStrategy(ShardingStrategy):
    """Pipeline Parallelism: assign sequential layer groups to different GPUs.

    GPU 0 gets layers 0..L/K, GPU 1 gets L/K..2L/K, etc.
    For inference (no backward pass), this is a simple sequential forwarding
    through stages — no microbatch scheduling needed.
    """

    def compute_gpus_needed(self, model_vram_gb: float, gpu_vram_gb: float) -> int:
        # PP: each GPU holds ~1/K of the model + activations for its stage
        return math.ceil(model_vram_gb / gpu_vram_gb)

    def shard_model(self, adapter: ModelAdapter, device_ids: list[int]) -> Any:
        """Partition layer list across GPUs.

        Steps:
            1. Load model to CPU
            2. Extract layer list
            3. Partition into len(device_ids) groups
            4. Move each group to its assigned GPU
            5. Move embedding to first GPU, head to last GPU
        """
        # Load to CPU for partitioning
        adapter.load_to_device(torch.device("cpu"))
        model = adapter._active_model  # type: ignore[attr-defined]
        if model is None:
            raise ShardingError("Adapter model not loaded after load_to_device().")

        layers, embedding, head = self._decompose_model(model)

        if not layers:
            raise ShardingError(
                "Could not find sequential layers in model for pipeline parallelism. "
                "Model must have a 'layers', 'blocks', or 'transformer.h' attribute."
            )

        num_stages = len(device_ids)
        if len(layers) < num_stages:
            raise ShardingError(
                f"Model has {len(layers)} layers but {num_stages} pipeline stages requested."
            )

        # Partition layers across stages
        chunk_size = len(layers) // num_stages
        remainder = len(layers) % num_stages
        stages: list[_PPStage] = []
        offset = 0

        for stage_idx in range(num_stages):
            # Distribute remainder evenly across first stages
            n = chunk_size + (1 if stage_idx < remainder else 0)
            stage_layers = nn.Sequential(*layers[offset : offset + n])
            device = torch.device(f"cuda:{device_ids[stage_idx]}")
            stage_layers.to(device)
            stages.append(_PPStage(stage_layers, device))
            offset += n

        first_device = torch.device(f"cuda:{device_ids[0]}")
        last_device = torch.device(f"cuda:{device_ids[-1]}")

        # Move embedding to first device, head to last
        if embedding is not None:
            embedding.to(first_device)
        if head is not None:
            head.to(last_device)

        _log.info(
            "Applied PP: %d layers split into %d stages across GPUs %s",
            len(layers),
            num_stages,
            device_ids,
        )

        return _PPModelWrapper(stages, embedding, head, adapter.eval_fn, first_device, last_device)

    def run_sharded(self, sharded_model: Any, batch: Any) -> Any:
        wrapper: _PPModelWrapper = sharded_model

        with torch.no_grad():
            # Move input to first device
            if isinstance(batch, torch.Tensor):
                x = batch.to(wrapper.first_device, non_blocking=True)
            elif isinstance(batch, dict):
                x = {
                    k: v.to(wrapper.first_device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                x = batch

            # Embedding stage
            if wrapper.embedding is not None:
                if isinstance(x, dict):
                    # For transformer models, typically pass input_ids through embedding
                    input_ids = x.get("input_ids", x.get("input", None))
                    if input_ids is not None:
                        x = wrapper.embedding(input_ids)
                    else:
                        x = wrapper.embedding(x)
                else:
                    x = wrapper.embedding(x)

            # Forward through pipeline stages
            for stage in wrapper.stages:
                if isinstance(x, torch.Tensor):
                    x = x.to(stage.device, non_blocking=True)
                x = stage.layers(x)

            # Head stage
            if wrapper.head is not None:
                if isinstance(x, torch.Tensor):
                    x = x.to(wrapper.last_device, non_blocking=True)
                x = wrapper.head(x)

            # Move result to CPU
            if isinstance(x, torch.Tensor):
                x = x.cpu()

            return x

    @staticmethod
    def _decompose_model(model: nn.Module) -> tuple[list[nn.Module], nn.Module | None, nn.Module | None]:
        """Extract layers, embedding, and head from a model.

        Tries common transformer naming patterns.
        Returns (layers_list, embedding_or_None, head_or_None).
        """
        layers: list[nn.Module] = []
        embedding: nn.Module | None = None
        head: nn.Module | None = None

        # Try common layer list attributes
        for attr in ("layers", "blocks", "h"):
            layer_list = getattr(model, attr, None)
            if layer_list is not None and isinstance(layer_list, nn.ModuleList):
                layers = list(layer_list)
                break

        # Try transformer.h pattern (GPT-2 style)
        if not layers:
            transformer = getattr(model, "transformer", None)
            if transformer is not None:
                for attr in ("h", "layers", "blocks"):
                    layer_list = getattr(transformer, attr, None)
                    if layer_list is not None and isinstance(layer_list, nn.ModuleList):
                        layers = list(layer_list)
                        break

        # Try to find embedding layer
        for attr in ("embed_tokens", "wte", "embedding", "token_emb", "tok_emb"):
            emb = getattr(model, attr, None)
            if emb is None:
                transformer = getattr(model, "transformer", None)
                if transformer:
                    emb = getattr(transformer, attr, None)
            if isinstance(emb, nn.Module):
                embedding = emb
                break

        # Try to find output head
        for attr in ("lm_head", "head", "output", "classifier", "fc_out"):
            hd = getattr(model, attr, None)
            if isinstance(hd, nn.Module):
                head = hd
                break

        return layers, embedding, head
