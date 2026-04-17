from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from octopus.adapters.base import ModelAdapter


class ShardingStrategy(ABC):
    """Abstract base for model sharding across multiple GPUs."""

    @abstractmethod
    def compute_gpus_needed(self, model_vram_gb: float, gpu_vram_gb: float) -> int:
        """How many GPUs are needed to host this model."""
        ...

    @abstractmethod
    def shard_model(self, adapter: ModelAdapter, device_ids: list[int]) -> Any:
        """Apply sharding across the given devices. Returns a sharded model handle."""
        ...

    @abstractmethod
    def run_sharded(self, sharded_model: Any, batch: Any) -> Any:
        """Run inference on the sharded model."""
        ...
