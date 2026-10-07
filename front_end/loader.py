"""Imports an ONNX graph as an `hc` module with one `func.func`.

Types come from ONNX shapes: rank 0 -> i32, rank 1 to 3 -> TensorType. Graph
inputs use their declared shape (`_type_from_value_info`); initializers and
`Constant` nodes use the same rule (`_const_op_from_tensor`), so a constant
and a runtime input of the same shape get the same type. A value that isn't
INT32, has rank 4 or more, or has a dim without a known positive size is
rejected (`_check_value`).

Each node picks its hc op variant (scalar or `*_tensor`) from the types of
the operands it receives. `Add`, `Sub` and `Mul` take the tensor variant when
either operand is a tensor; its result type is the shape the operands
broadcast to (`broadcast_shape`), so the other operand may be a scalar, have
a lower rank, or have dims of size 1. The op is verified as it is built, so
operands that don't fit (for example a tensor given to `Pow`) are rejected
with the node's name.
"""
import onnx
from xdsl.dialects import func, arith, builtin

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Region, Block
from xdsl.utils.exceptions import VerifyException

from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu, HCPow, HCMax, HCMin
from hc_dialect import HCMatmul, HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor
from hc_dialect import broadcast_shape, shape_of

# --------------------------------------
#  Helper functions
# --------------------------------------

def _check_value(name: str, elem_type: int, dims: list) -> None:
    """Raises NotImplementedError unless the ONNX value `name` is INT32 with
    rank 0-3 and every dim a known positive size."""
    if elem_type != onnx.TensorProto.INT32:
        raise NotImplementedError(
            f"ONNX value {name!r}: element type "
            f"{onnx.TensorProto.DataType.Name(elem_type)}, only INT32 is supported"
        )
    if len(dims) > 3:
        raise NotImplementedError(
            f"ONNX value {name!r}: rank {len(dims)}, only ranks 0-3 are supported"
        )
    if any(not isinstance(d, int) or d <= 0 for d in dims):
        raise NotImplementedError(
            f"ONNX value {name!r}: shape {dims}, every dim must be a known positive size"
        )


def _type_from_value_info(value_info):
    """Returns the hc type for an ONNX graph input's declared shape."""
    tensor_type = value_info.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in tensor_type.shape.dim]
    _check_value(value_info.name, tensor_type.elem_type, dims)
    if not dims:
        return i32
    return builtin.TensorType(i32, dims)


def _const_op_from_tensor(tensor_proto, name: str) -> arith.ConstantOp:
    """Builds an `arith.constant` (scalar i32, or dense tensor) from an ONNX
    TensorProto bound to the value `name`."""
    _check_value(name, tensor_proto.data_type, list(tensor_proto.dims))
    arr = onnx.numpy_helper.to_array(tensor_proto)
    if arr.shape == ():
        return arith.ConstantOp.from_int_and_width(int(arr), 32)
    ty = builtin.TensorType(i32, list(arr.shape))
    values = [int(v) for v in arr.reshape(-1).tolist()]
    dense = builtin.DenseIntOrFPElementsAttr.from_list(ty, values)
    return arith.ConstantOp(dense)


def _is_tensor(value) -> bool:
    return isinstance(value.type, builtin.TensorType)


def _dim_as_int(int_attr) -> int:
    return int_attr.data


# --------------------------------------
#  Loader
# --------------------------------------
def import_onnx_to_hc_module(
    ctx: Context, onnx_path: str, fn_name: str = "main"
) -> ModuleOp:
    """Loads `onnx_path` and returns a module holding `func.func @fn_name`,
    with one block argument per graph input and a single result.

    Raises NotImplementedError for an unsupported op_type, element type, rank
    or dim, or operand types the node's hc op rejects (for example a tensor
    given to Pow); ValueError for operand shapes that don't broadcast, a model
    without exactly one output or a `Constant` node without a `value`
    attribute, and KeyError when a node reads a value that is not yet
    defined.
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # Map ONNX value names -> SSAValue (results)
    env: dict[str, object] = {}

    # One block argument per ONNX graph input, typed from its declared shape.
    input_types = [_type_from_value_info(vi) for vi in graph.input]
    entry_block = Block(arg_types=input_types)
    for value_info, arg in zip(graph.input, entry_block.args):
        env[value_info.name] = arg

    ops = []

    # 1) Initializers (weights/constants stored in graph.initializer)
    for init in graph.initializer:
        c = _const_op_from_tensor(init, init.name)
        ops.append(c)
        env[init.name] = c.result

    # 2) Constant nodes (op_type == "Constant")
    # ONNX Constant usually stores a tensor in attribute "value"
    def emit_constant_node(node):
        value_attr = None
        for a in node.attribute:
            if a.name == "value":
                value_attr = a.t
        if value_attr is None:
            raise ValueError("Constant node without 'value' attribute")
        c = _const_op_from_tensor(value_attr, node.output[0])
        ops.append(c)
        env[node.output[0]] = c.result

    # 3) Node lowering
    for node in graph.node:
        if node.op_type == "Constant":
            emit_constant_node(node)
            continue

        def get(name: str):
            if name not in env:
                raise KeyError(f"ONNX value not found yet: {name} (node {node.op_type})")
            return env[name]

        def emit(hc_op):
            """Verifies `hc_op`, naming this node if its operands don't fit,
            then binds its result to the node's output."""
            try:
                hc_op.verify()
            except VerifyException as e:
                types = ", ".join(str(v.type) for v in hc_op.operands)
                raise NotImplementedError(
                    f"ONNX {node.op_type} node {node.name or node.output[0]!r}: "
                    f"{hc_op.name} rejects operand types ({types})"
                ) from e
            ops.append(hc_op)
            env[node.output[0]] = hc_op.results[0]

        def broadcast_type(a, b):
            """Returns the tensor type `a` and `b` broadcast to, naming this
            node if their shapes don't fit."""
            try:
                shape = broadcast_shape(shape_of(a.type), shape_of(b.type))
            except ValueError as e:
                raise ValueError(
                    f"ONNX {node.op_type} node {node.name or node.output[0]!r}: {e}"
                ) from e
            return builtin.TensorType(i32, shape)

        if node.op_type == "Add":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_tensor(a) or _is_tensor(b):
                hc_cls, res_ty = HCAddTensor, broadcast_type(a, b)
            else:
                hc_cls, res_ty = HCAdd, a.type
            hc = hc_cls(operands=[a, b], result_types=[res_ty])
            emit(hc)
            continue

        if node.op_type == "Sub":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_tensor(a) or _is_tensor(b):
                hc_cls, res_ty = HCSubTensor, broadcast_type(a, b)
            else:
                hc_cls, res_ty = HCSub, a.type
            hc = hc_cls(operands=[a, b], result_types=[res_ty])
            emit(hc)
            continue

        if node.op_type == "Mul":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_tensor(a) or _is_tensor(b):
                hc = HCMulTensor(operands=[a, b], result_types=[broadcast_type(a, b)])
            else:
                hc = HCMul(operands=[a, b], result_types=[i32])
            emit(hc)
            continue

        if node.op_type == "MatMul":
            a = get(node.input[0])
            b = get(node.input[1])
            a_dims = [_dim_as_int(d) for d in a.type.shape]
            b_dims = [_dim_as_int(d) for d in b.type.shape]
            batch, m = a_dims[:-2], a_dims[-2]
            n = b_dims[-1]
            res_ty = builtin.TensorType(i32, batch + [m, n])
            hc = HCMatmul(operands=[a, b], result_types=[res_ty])
            emit(hc)
            continue

        if node.op_type == "Relu":
            x = get(node.input[0])
            hc_cls = HCReluTensor if _is_tensor(x) else HCRelu
            hc = hc_cls(operands=[x], result_types=[x.type])
            emit(hc)
            continue

        if node.op_type == "Pow":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_pow = HCPow(operands=[a, b], result_types=[i32])
            emit(hc_pow)
            continue

        if node.op_type == "Max":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_max = HCMax(operands=[a, b], result_types=[i32])
            emit(hc_max)
            continue

        if node.op_type == "Min":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_min = HCMin(operands=[a, b], result_types=[i32])
            emit(hc_min)
            continue

        raise NotImplementedError(f"Unsupported ONNX op: {node.op_type}")

    # 4) Determine outputs (graph.output)
    if len(graph.output) != 1:
        raise ValueError("For now: expect exactly 1 model output")
    out_name = graph.output[0].name
    if out_name not in env:
        raise KeyError(f"Graph output {out_name} not produced")
    ret = func.ReturnOp(env[out_name])
    ops.append(ret)

    for op in ops:
        entry_block.add_op(op)

    out_type = env[out_name].type
    fn = func.FuncOp(fn_name, (input_types, [out_type]), region=Region(entry_block))

    return ModuleOp(ops=[fn])
