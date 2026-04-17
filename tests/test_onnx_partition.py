import pytest

from octopus.exceptions import ShardingError
from octopus.sharding.onnx_partition import (
    infer_stage_io,
    materialize_stage_models,
    partition_node_sequence,
)


def test_partition_node_sequence_balanced_and_contiguous():
    node_names = [f"n{i}" for i in range(10)]

    plans = partition_node_sequence(node_names, num_stages=3)

    assert [p.stage_id for p in plans] == [0, 1, 2]
    assert [len(p.node_names) for p in plans] == [4, 3, 3]
    assert list(plans[0].node_names + plans[1].node_names + plans[2].node_names) == node_names


def test_partition_node_sequence_rejects_more_stages_than_nodes():
    with pytest.raises(ShardingError, match="Cannot split"):
        partition_node_sequence(["n0", "n1"], num_stages=3)


def test_partition_node_sequence_rejects_duplicate_node_names():
    with pytest.raises(ValueError, match="unique"):
        partition_node_sequence(["n0", "n0", "n1"], num_stages=2)


def test_partition_node_sequence_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="num_stages"):
        partition_node_sequence(["n0"], num_stages=0)

    with pytest.raises(ValueError, match="must not be empty"):
        partition_node_sequence([], num_stages=1)


def test_infer_stage_io_for_simple_chain(tmp_path):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper

    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    c1 = helper.make_tensor("c1", onnx.TensorProto.FLOAT, [1], [1.0])
    c2 = helper.make_tensor("c2", onnx.TensorProto.FLOAT, [1], [2.0])
    n0 = helper.make_node("Add", ["x", "c1"], ["z0"], name="n0")
    n1 = helper.make_node("Mul", ["z0", "c2"], ["y"], name="n1")
    graph = helper.make_graph([n0, n1], "g", [x], [y], initializer=[c1, c2])
    model = helper.make_model(graph)

    plans = partition_node_sequence(["n0", "n1"], num_stages=2)
    ios = infer_stage_io(model, plans)

    assert ios[0][0] == ("x",)
    assert ios[0][1] == ("z0",)
    assert ios[1][0] == ("z0",)
    assert ios[1][1] == ("y",)


def test_materialize_stage_models_for_simple_chain(tmp_path):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper

    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    c1 = helper.make_tensor("c1", onnx.TensorProto.FLOAT, [1], [1.0])
    c2 = helper.make_tensor("c2", onnx.TensorProto.FLOAT, [1], [2.0])
    n0 = helper.make_node("Add", ["x", "c1"], ["z0"], name="n0")
    n1 = helper.make_node("Mul", ["z0", "c2"], ["y"], name="n1")
    graph = helper.make_graph([n0, n1], "g", [x], [y], initializer=[c1, c2])
    model = helper.make_model(graph)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    plans = partition_node_sequence(["n0", "n1"], num_stages=2)
    artifacts = materialize_stage_models(
        str(model_path),
        plans,
        output_dir=str(tmp_path / "stages"),
    )

    assert len(artifacts) == 2
    assert artifacts[0].input_names == ("x",)
    assert artifacts[0].output_names == ("z0",)
    assert artifacts[1].input_names == ("z0",)
    assert artifacts[1].output_names == ("y",)
    assert (tmp_path / "stages" / "stage_0.onnx").exists()
    assert (tmp_path / "stages" / "stage_1.onnx").exists()
