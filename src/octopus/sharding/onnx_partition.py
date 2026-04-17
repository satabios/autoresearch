from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from octopus.exceptions import ShardingError


@dataclass(frozen=True)
class OnnxStagePlan:
    """One contiguous ONNX node slice assigned to one pipeline stage."""

    stage_id: int
    node_names: tuple[str, ...]


@dataclass(frozen=True)
class OnnxStageArtifact:
    """Materialized ONNX stage model and explicit IO contract."""

    stage_id: int
    onnx_path: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    node_names: tuple[str, ...]


def partition_node_sequence(
    node_names: Sequence[str],
    num_stages: int,
) -> list[OnnxStagePlan]:
    """Split ONNX node sequence into contiguous stages.

    First implementation is intentionally simple:
    contiguous balanced chunks by node count.
    """
    if num_stages <= 0:
        raise ValueError("num_stages must be >= 1")
    if not node_names:
        raise ValueError("node_names must not be empty")

    normalized = [str(name) for name in node_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("node_names must be unique")

    total_nodes = len(normalized)
    if num_stages > total_nodes:
        raise ShardingError(
            f"Cannot split {total_nodes} ONNX node(s) into {num_stages} stage(s)."
        )

    base = total_nodes // num_stages
    remainder = total_nodes % num_stages

    plans: list[OnnxStagePlan] = []
    offset = 0
    for stage_id in range(num_stages):
        size = base + (1 if stage_id < remainder else 0)
        stage_nodes = tuple(normalized[offset : offset + size])
        plans.append(OnnxStagePlan(stage_id=stage_id, node_names=stage_nodes))
        offset += size

    return plans


def load_node_names_from_onnx(onnx_path: str) -> list[str]:
    """Load node names from ONNX model path."""
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "onnx package required for ONNX partition planning. "
            "Install with pip install onnx."
        ) from e

    model = onnx.load(onnx_path)
    names: list[str] = []
    for idx, node in enumerate(model.graph.node):
        if node.name:
            names.append(node.name)
        else:
            names.append(f"{node.op_type}_{idx}")
    return names


def plan_stages_from_onnx(onnx_path: str, num_stages: int) -> list[OnnxStagePlan]:
    """Convenience wrapper: load ONNX nodes then partition."""
    node_names = load_node_names_from_onnx(onnx_path)
    return partition_node_sequence(node_names, num_stages=num_stages)


def materialize_stage_models(
    onnx_path: str,
    stage_plans: Sequence[OnnxStagePlan],
    output_dir: str,
) -> list[OnnxStageArtifact]:
    """Export stage ONNX files using computed stage boundaries."""
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "onnx package required for ONNX stage materialization. "
            "Install with pip install onnx."
        ) from e

    model = onnx.load(onnx_path)
    stage_ios = infer_stage_io(model, stage_plans)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts: list[OnnxStageArtifact] = []
    for stage_plan, (stage_inputs, stage_outputs) in zip(stage_plans, stage_ios):
        stage_path = output_root / f"stage_{stage_plan.stage_id}.onnx"
        onnx.utils.extract_model(  # type: ignore[attr-defined]
            onnx_path,
            str(stage_path),
            list(stage_inputs),
            list(stage_outputs),
        )
        artifacts.append(
            OnnxStageArtifact(
                stage_id=stage_plan.stage_id,
                onnx_path=str(stage_path),
                input_names=tuple(stage_inputs),
                output_names=tuple(stage_outputs),
                node_names=tuple(stage_plan.node_names),
            )
        )

    return artifacts


def infer_stage_io(
    model: object,
    stage_plans: Sequence[OnnxStagePlan],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Infer stage input/output tensors from ONNX graph boundaries."""
    graph = model.graph  # type: ignore[attr-defined]
    graph_inputs = {value.name for value in graph.input}
    graph_outputs = {value.name for value in graph.output}
    initializers = {value.name for value in graph.initializer}

    node_by_name = {node.name: node for node in graph.node}
    if len(node_by_name) != len(graph.node):
        raise ShardingError(
            "ONNX graph has duplicate or empty node names. "
            "Use load_node_names_from_onnx synthesized names for planning."
        )

    producer_by_tensor: dict[str, str] = {}
    consumers_by_tensor: dict[str, list[str]] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer_by_tensor[out] = node.name
        for inp in node.input:
            if inp:
                consumers_by_tensor.setdefault(inp, []).append(node.name)

    tensor_order: dict[str, int] = {}
    order_index = 0
    for node in graph.node:
        for tensor_name in (*node.input, *node.output):
            if tensor_name and tensor_name not in tensor_order:
                tensor_order[tensor_name] = order_index
                order_index += 1

    stage_io: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for plan in stage_plans:
        stage_set = set(plan.node_names)
        stage_inputs: set[str] = set()
        stage_outputs: set[str] = set()

        for node_name in plan.node_names:
            node = node_by_name.get(node_name)
            if node is None:
                raise ShardingError(f"Stage references missing node: {node_name}")

            for inp in node.input:
                if not inp or inp in initializers:
                    continue
                producer = producer_by_tensor.get(inp)
                if producer is None:
                    if inp in graph_inputs:
                        stage_inputs.add(inp)
                    continue
                if producer not in stage_set:
                    stage_inputs.add(inp)

            for out in node.output:
                if not out:
                    continue
                consumers = consumers_by_tensor.get(out, [])
                has_outside_consumer = any(c not in stage_set for c in consumers)
                if has_outside_consumer or out in graph_outputs:
                    stage_outputs.add(out)

        if not stage_outputs:
            # last stage may rely only on graph outputs present in plan;
            # if still empty, exported subgraph would be invalid.
            raise ShardingError(
                f"Stage {plan.stage_id} has no boundary outputs; cannot materialize."
            )

        ordered_inputs = tuple(sorted(stage_inputs, key=lambda n: tensor_order.get(n, 10**9)))
        ordered_outputs = tuple(sorted(stage_outputs, key=lambda n: tensor_order.get(n, 10**9)))
        stage_io.append((ordered_inputs, ordered_outputs))

    return stage_io
