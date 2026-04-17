from __future__ import annotations

from typing import Literal

from octopus.sharding.base import ShardingStrategy
from octopus.sharding.onnx_partition import (
    OnnxStageArtifact,
    OnnxStagePlan,
    infer_stage_io,
    load_node_names_from_onnx,
    materialize_stage_models,
    partition_node_sequence,
    plan_stages_from_onnx,
)
from octopus.sharding.onnx_pipeline_runtime import (
    ORTPipelineSessionProxy,
    build_onnx_pipeline_proxy,
    normalize_ort_input_feed,
)
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


__all__ = [
    "ShardingStrategy",
    "TensorParallelStrategy",
    "PipelineParallelStrategy",
    "OnnxStagePlan",
    "OnnxStageArtifact",
    "partition_node_sequence",
    "load_node_names_from_onnx",
    "plan_stages_from_onnx",
    "infer_stage_io",
    "materialize_stage_models",
    "ORTPipelineSessionProxy",
    "normalize_ort_input_feed",
    "build_onnx_pipeline_proxy",
    "get_strategy",
]
