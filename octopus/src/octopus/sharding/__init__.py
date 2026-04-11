from __future__ import annotations

from typing import Literal

from octopus.sharding.base import ShardingStrategy
from octopus.sharding.pipeline_parallel import PipelineParallelStrategy
from octopus.sharding.tensor_parallel import TensorParallelStrategy

_STRATEGIES: dict[str, type[ShardingStrategy]] = {
    "tp": TensorParallelStrategy,
    "pp": PipelineParallelStrategy,
}


def get_strategy(name: Literal["tp", "pp"]) -> ShardingStrategy:
    """Return an instantiated sharding strategy by name."""
    cls = _STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown sharding strategy: {name!r}. Choose from: {list(_STRATEGIES)}")
    return cls()
