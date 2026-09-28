"""ONNX -> hc op mapping: pins the scalar/vector dispatch logic in
import_onnx_to_hc_module -- given a declared shape (scalar [] vs vector [n]),
does each ONNX op_type map to the right hc op?
"""
import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto  # noqa: E402

from front_end.loader import import_onnx_to_hc_module  # noqa: E402
from conftest import entry_op_names  # noqa: E402


def _save_one_node_model(tmp_path, op_type, input_specs, output_shape, name="f"):
    """input_specs: list of (name, shape) -- shape=[] for scalar, [n] for vector."""
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
    ("Add", [4], "hc.add_vec"),
    ("Sub", [4], "hc.sub_vec"),
    ("Relu", [4], "hc.relu_vec"),
    ("Add", [2, 3], "hc.add_tensor"),
    ("Sub", [2, 3], "hc.sub_tensor"),
    ("Relu", [2, 3], "hc.relu_tensor"),
    ("Add", [2, 3, 4], "hc.add_tensor"),
    ("Sub", [2, 3, 4], "hc.sub_tensor"),
    ("Relu", [2, 3, 4], "hc.relu_tensor"),
])
def test_binop_or_unary_dispatches_by_shape(tmp_path, ctx, op_type, shape, expected_hc_op):
    if op_type == "Relu":
        specs = [("x", shape)]
    else:
        specs = [("a", shape), ("b", shape)]
    path = _save_one_node_model(tmp_path, op_type, specs, shape)
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert expected_hc_op in entry_op_names(module, "my_func")


def test_mul_vecvec_dispatches_to_mul_vec_vec(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", [4]), ("b", [4])], [4])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.mul_vec_vec" in entry_op_names(module, "my_func")


def test_mul_scalar_times_vector_dispatches_to_mul_vec(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", []), ("b", [4])], [4])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.mul_vec" in entry_op_names(module, "my_func")


def test_mul_tensor_dispatches_to_mul_tensor(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", [2, 3]), ("b", [2, 3])], [2, 3])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.mul_tensor" in entry_op_names(module, "my_func")


def test_batched_mul_tensor_dispatches_to_mul_tensor(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Mul", [("a", [2, 3, 4]), ("b", [2, 3, 4])], [2, 3, 4])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")
    assert "hc.mul_tensor" in entry_op_names(module, "my_func")


def test_multiple_inputs_all_bound_as_block_args(tmp_path, ctx):
    """Regression for the old 'assume one scalar input' limitation."""
    path = _save_one_node_model(tmp_path, "Add", [("a", []), ("b", [])], [])
    module = import_onnx_to_hc_module(ctx, path, fn_name="my_func")

    from xdsl.dialects import func
    fn = next(
        op for block in module.body.blocks for op in block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "my_func"
    )
    assert len(fn.function_type.inputs.data) == 2


def test_pow_rejects_vector_operand(tmp_path, ctx):
    path = _save_one_node_model(tmp_path, "Pow", [("a", [4]), ("b", [4])], [4])
    with pytest.raises(NotImplementedError):
        import_onnx_to_hc_module(ctx, path, fn_name="my_func")


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
