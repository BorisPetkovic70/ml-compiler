"""ONNX -> hc.* module: one loader, dispatching on declared/observed shape.

Each ONNX graph *input*'s hc type is decided once, from its declared shape alone
(_type_from_value_info: rank-0 -> i32, rank-2 -> TensorType, else -> VectorType).
Each *node* then picks its concrete hc.* op class (scalar vs. *_vec vs. *_tensor
vs. matmul) from the operand types it actually sees at that point in the graph --
not from the node's own declared shape, since ONNX doesn't attach one to
intermediate values the way it does to graph inputs.

ONNX *initializers*/`Constant` nodes go through a separate function,
_const_op_from_tensor, deliberately mirroring the same rank-2 rule -- a constant
and a graph input of the same shape must produce the same hc type, or a
downstream op (hc.matmul) would see mismatched operand types depending on
whether one operand came from an initializer or a runtime input.

See docs/DESIGN.md Section 1 for why this dispatch happens once, in this file,
rather than being re-derived at every consumer downstream.
"""
import onnx
from xdsl.dialects import func, arith, builtin

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Region, Block

from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu, HCPow, HCMax, HCMin
from hc_dialect import HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec
from hc_dialect import HCMatmul, HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor

# --------------------------------------
#  Helper functions
# --------------------------------------

def _vec_type_from_shape(shape) -> builtin.VectorType:
    """Build a VectorType<...xi32> from a list of dimension sizes."""
    dims = [int(d) for d in shape]
    try:
        return builtin.VectorType(dims, i32)
    except TypeError:
        # some xdsl versions take (element_type, shape)
        return builtin.VectorType(i32, dims)


def _type_from_value_info(value_info):
    """i32 for a rank-0 (scalar) ONNX input, VectorType for rank-1, TensorType
    for rank-2 (matmul operands)."""
    dims = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
    if not dims:
        return i32
    if len(dims) == 2:
        return builtin.TensorType(i32, dims)
    return _vec_type_from_shape(dims)


def _const_op_from_tensor(tensor_proto) -> arith.ConstantOp:
    """arith.constant from an ONNX tensor: scalar i32 if rank-0, dense tensor if
    rank-2 (matmul operands), dense vector otherwise."""
    arr = onnx.numpy_helper.to_array(tensor_proto)
    if arr.shape == ():
        return arith.ConstantOp.from_int_and_width(int(arr), 32)
    if len(arr.shape) == 2:
        ty = builtin.TensorType(i32, list(arr.shape))
    else:
        ty = _vec_type_from_shape(arr.shape)
    values = [int(v) for v in arr.reshape(-1).tolist()]
    dense = builtin.DenseIntOrFPElementsAttr.from_list(ty, values)
    return arith.ConstantOp(dense)


def _is_vec(value) -> bool:
    return isinstance(value.type, builtin.VectorType)


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
    model = onnx.load(onnx_path)
    graph = model.graph

    # Map ONNX value names -> SSAValue (results)
    env: dict[str, object] = {}

    # One block argument per ONNX graph input; scalar (i32) or vector,
    # depending on that input's declared shape.
    input_types = [_type_from_value_info(vi) for vi in graph.input]
    entry_block = Block(arg_types=input_types)
    for value_info, arg in zip(graph.input, entry_block.args):
        env[value_info.name] = arg

    ops = []

    # 1) Initializers (weights/constants stored in graph.initializer)
    for init in graph.initializer:
        c = _const_op_from_tensor(init)
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
        c = _const_op_from_tensor(value_attr)
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

        if node.op_type == "Add":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_tensor(a) and _is_tensor(b):
                hc_cls = HCAddTensor
            elif _is_vec(a) and _is_vec(b):
                hc_cls = HCAddVec
            else:
                hc_cls = HCAdd
            hc = hc_cls(operands=[a, b], result_types=[a.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Sub":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_tensor(a) and _is_tensor(b):
                hc_cls = HCSubTensor
            elif _is_vec(a) and _is_vec(b):
                hc_cls = HCSubVec
            else:
                hc_cls = HCSub
            hc = hc_cls(operands=[a, b], result_types=[a.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Mul":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_tensor(a) and _is_tensor(b):
                hc = HCMulTensor(operands=[a, b], result_types=[a.type])
            elif _is_vec(a) and _is_vec(b):
                hc = HCMulVecVec(operands=[a, b], result_types=[a.type])
            elif _is_vec(a) or _is_vec(b):
                # hc.mul_vec is scalar * vector: put the vector operand second.
                scalar, vec = (b, a) if _is_vec(a) else (a, b)
                hc = HCMulVec(operands=[scalar, vec], result_types=[vec.type])
            else:
                hc = HCMul(operands=[a, b], result_types=[i32])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "MatMul":
            a = get(node.input[0])
            b = get(node.input[1])
            m = _dim_as_int(list(a.type.shape)[0])
            n = _dim_as_int(list(b.type.shape)[1])
            res_ty = builtin.TensorType(i32, [m, n])
            hc = HCMatmul(operands=[a, b], result_types=[res_ty])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Relu":
            x = get(node.input[0])
            if _is_tensor(x):
                hc_cls = HCReluTensor
            elif _is_vec(x):
                hc_cls = HCReluVec
            else:
                hc_cls = HCRelu
            hc = hc_cls(operands=[x], result_types=[x.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Pow":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_vec(a) or _is_vec(b):
                raise NotImplementedError("hc.pow has no vector variant yet")
            hc_pow = HCPow(operands=[a, b], result_types=[i32])
            ops.append(hc_pow)
            env[node.output[0]] = hc_pow.results[0]
            continue

        if node.op_type == "Max":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_vec(a) or _is_vec(b):
                raise NotImplementedError("hc.max has no vector variant yet")
            hc_max = HCMax(operands=[a, b], result_types=[i32])
            ops.append(hc_max)
            env[node.output[0]] = hc_max.results[0]
            continue

        if node.op_type == "Min":
            a = get(node.input[0])
            b = get(node.input[1])
            if _is_vec(a) or _is_vec(b):
                raise NotImplementedError("hc.min has no vector variant yet")
            hc_min = HCMin(operands=[a, b], result_types=[i32])
            ops.append(hc_min)
            env[node.output[0]] = hc_min.results[0]
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
