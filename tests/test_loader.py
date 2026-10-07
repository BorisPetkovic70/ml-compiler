"""ONNX -> hc op mapping: given the declared input shapes (scalar [], or a
tensor of rank 1 to 3), each ONNX op_type maps to the right hc op variant.
"""
import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto  # noqa: E402

from front_end.loader import import_onnx_to_hc_module  # noqa: E402
from conftest import entry_op_names, find_op  # noqa: E402


def _save_one_node_model(tmp_path, op_type, input_specs, output_shape, name="f"):
    """Saves a one-node INT32 model and returns its path. `input_specs` is a
    list of (name, shape)."""
    inputs = [helper.make_tensor_value_info(n, TensorProto.INT32, s) for n, s in input_specs]
    output = helper.make_tensor_value_info("y", TensorProto.INT32, output_shape)
    node = helper.make_node(op_type, [n for n, _ in input_specs], ["y"], name=name)
    graph = helper.make_graph(nodes=[node], name="g", inputs=inputs, outputs=[output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.mark.parametrize("op_type,shape,expected_hc_op", [
    ("Add", [], "hc.add"),
    ("Sub", [], "hc.sub"),
    ("Mul", [], "hc.mul"),
    ("Relu", [], "hc.relu"),
    ("Add", [4], "hc.add_tensor"),
    ("Sub", [4], "hc.sub_tensor"),
    ("Relu", [4], "hc.relu_tensor"),
    ("Add", [2, 3], "hc.add_tensor"),
    ("Sub", [2, 3], "hc.sub_tensor"),
    ("Relu", [2, 3], "hc.relu_tensor"),
    ("Add", [2, 3, 4], "hc.add_tensor"),
])
def test_binop_or_unary_dispatches_by_shape(tmp_path, ctx, op_type, shape, expected_hc_op):
    if op_type == "Relu":
        specs = [("x", shape)]
    else:
        specs = [("a", shape), ("b", shape)]
    path = _save_one_node_model(tmp_path, op_type, specs, shape)
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert expected_hc_op in entry_op_names(module, "my_func")


def test_mul_rank_1_dispatches_to_mul_tensor(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", [4]), ("b", [4])], [4])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.mul_tensor" in entry_op_names(module, "my_func")


def test_mul_scalar_times_rank_1_dispatches_to_mul_tensor(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", []), ("b", [4])], [4])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    op = find_op(module, "hc.mul_tensor", "my_func")
    assert [d.data for d in op.results[0].type.shape] == [4]
    module.verify()


def test_mul_tensor_dispatches_to_mul_tensor(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", [2, 3]), ("b", [2, 3])], [2, 3])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.mul_tensor" in entry_op_names(module, "my_func")


def test_add_result_type_is_the_broadcast_shape(tmp_path, ctx):
    """1x3 + 2x3 -> 2x3: the result type is not the first operand's."""
    path = _save_one_node_model(tmp_path, "Add", [("a", [1, 3]), ("b", [2, 3])], [2, 3])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    op = find_op(module, "hc.add_tensor", "my_func")
    assert [d.data for d in op.results[0].type.shape] == [2, 3]
    module.verify()


def test_bias_add_broadcasts_a_rank_1_operand(tmp_path, ctx):
    """2x3 + 3: a rank-1 value is a tensor, so it broadcasts along the rows."""
    path = _save_one_node_model(tmp_path, "Add", [("a", [2, 3]), ("b", [3])], [2, 3])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    op = find_op(module, "hc.add_tensor", "my_func")
    assert [d.data for d in op.results[0].type.shape] == [2, 3]
    module.verify()


def test_mul_scalar_times_tensor_dispatches_to_mul_tensor(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", []), ("b", [2, 3])], [2, 3])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    op = find_op(module, "hc.mul_tensor", "my_func")
    assert [d.data for d in op.results[0].type.shape] == [2, 3]
    module.verify()


def test_multiple_inputs_all_bound_as_block_args(tmp_path, ctx):
    """Every graph input becomes a function argument, not only the first."""
    path = _save_one_node_model(tmp_path, "Add", [("a", []), ("b", [])], [])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")

    from xdsl.dialects import func
    fn = next(
        op for block in module.body.blocks for op in block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "my_func"
    )
    assert len(fn.function_type.inputs.data) == 2


@pytest.mark.parametrize("op_type,expected_hc_op", [
    ("Pow", "hc.pow"), ("Max", "hc.max"), ("Min", "hc.min"),
])
def test_pow_max_min_take_tensors(tmp_path, ctx, op_type, expected_hc_op):
    """2x3 with 3: the result type is the broadcast shape."""
    path = _save_one_node_model(tmp_path, op_type, [("a", [2, 3]), ("b", [3])], [2, 3])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    op = find_op(module, expected_hc_op, "my_func")
    assert [d.data for d in op.results[0].type.shape] == [2, 3]
    module.verify()


def test_matmul_dispatches_to_hc_matmul(tmp_path, ctx):
    """MxK * KxN -> MxN, with M/K/N all distinct to catch a mixed-up dim."""
    a = helper.make_tensor_value_info("a", TensorProto.INT32, [4, 6])
    b = helper.make_tensor_value_info("b", TensorProto.INT32, [6, 8])
    c = helper.make_tensor_value_info("c", TensorProto.INT32, [4, 8])
    node = helper.make_node("MatMul", ["a", "b"], ["c"], name="mm")
    graph = helper.make_graph(nodes=[node], name="g", inputs=[a, b], outputs=[c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = str(tmp_path / "matmul.onnx")
    onnx.save(model, path)

    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.matmul" in entry_op_names(module, "my_func")

    from conftest import find_op
    op = find_op(module, "hc.matmul", "my_func")
    assert [d.data for d in op.results[0].type.shape] == [4, 8]


def test_batched_matmul_dispatches_to_hc_matmul(tmp_path, ctx):
    """BxMxK * BxKxN -> BxMxN, with B/M/K/N all distinct to catch a mixed-up dim."""
    a = helper.make_tensor_value_info("a", TensorProto.INT32, [2, 4, 6])
    b = helper.make_tensor_value_info("b", TensorProto.INT32, [2, 6, 8])
    c = helper.make_tensor_value_info("c", TensorProto.INT32, [2, 4, 8])
    node = helper.make_node("MatMul", ["a", "b"], ["c"], name="mm")
    graph = helper.make_graph(nodes=[node], name="g", inputs=[a, b], outputs=[c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = str(tmp_path / "batched_matmul.onnx")
    onnx.save(model, path)

    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.matmul" in entry_op_names(module, "my_func")

    from conftest import find_op
    op = find_op(module, "hc.matmul", "my_func")
    assert [d.data for d in op.results[0].type.shape] == [2, 4, 8]
    module.verify()


def test_matmul_with_rank2_initializer_operand_types_as_tensor(tmp_path, ctx):
    """A rank-2 ONNX initializer (a constant weight matrix, not a graph input) feeding
    hc.matmul must become a TensorType constant, not a VectorType one -- otherwise
    module.verify() rejects the operand ('should be of base attribute tensor')."""
    a = helper.make_tensor_value_info("a", TensorProto.INT32, [4, 2])
    out = helper.make_tensor_value_info("out", TensorProto.INT32, [4, 3])
    w_init = helper.make_tensor("w", TensorProto.INT32, [2, 3], list(range(6)))
    node = helper.make_node("MatMul", ["a", "w"], ["out"], name="mm")
    graph = helper.make_graph(
        nodes=[node], name="g", inputs=[a], outputs=[out], initializer=[w_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = str(tmp_path / "matmul_const.onnx")
    onnx.save(model, path)

    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    module.verify()  # would previously raise: vector<2x3xi32> should be of base attribute tensor

    from conftest import find_op
    const = find_op(module, "arith.constant", "my_func")
    assert [d.data for d in const.results[0].type.shape] == [2, 3]
    assert type(const.results[0].type).__name__ == "TensorType"
