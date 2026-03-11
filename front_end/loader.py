import onnx
from xdsl.dialects import func, arith

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Region, Block

from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu, HCPow, HCMax, HCMin
# --------------------------------------
#  Helper functions
# --------------------------------------
def _onnx_tensor_to_scalar_int(tensor_proto) -> int:
    arr = onnx.numpy_helper.to_array(tensor_proto)
    # Restriction: scalar or 1-element tensor
    if arr.size != 1:
        raise ValueError(f"Only scalar/1-element constants supported, got shape {arr.shape}")
    return int(arr.reshape(()))

def _make_i32_const(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(int(value), 32)

# --------------------------------------
#  Loader
# --------------------------------------
def import_onnx_to_hc_module(
    ctx: Context, onnx_path: str, fn_name: str="main"
) -> ModuleOp:
    model = onnx.load(onnx_path)
    graph = model.graph

    # Map ONNX value names -> SSAValue (results)
    env: dict[str, object] = {}

    # Assume one scalar input
    # Map ONNX graph input name to function/block argument
    entry_block = Block(arg_types=[i32])
    x_name = graph.input[0].name
    env[x_name] = entry_block.args[0]

    ops = []

    # 1) Initializers (weights/constants stored in graph.initializer)
    for init in graph.initializer:
        v = _onnx_tensor_to_scalar_int(init)
        c = _make_i32_const(v)
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
        v = _onnx_tensor_to_scalar_int(value_attr)
        c = _make_i32_const(v)
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
            hc_add = HCAdd(operands=[a, b], result_types=[i32])
            ops.append(hc_add)
            env[node.output[0]] = hc_add.results[0]
            continue

        if node.op_type == "Sub":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_sub = HCSub(operands=[a, b], result_types=[i32])
            ops.append(hc_sub)
            env[node.output[0]] = hc_sub.results[0]
            continue

        if node.op_type == "Mul":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_mul = HCMul(operands=[a, b], result_types=[i32])
            ops.append(hc_mul)
            env[node.output[0]] = hc_mul.results[0]
            continue

        if node.op_type == "Relu":
            x = get(node.input[0])
            hc_relu = HCRelu(operands=[x], result_types=[i32])
            ops.append(hc_relu)
            env[node.output[0]] = hc_relu.results[0]
            continue

        if node.op_type == "Pow":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_pow = HCPow(operands=[a, b], result_types=[i32])
            ops.append(hc_pow)
            env[node.output[0]] = hc_pow.results[0]
            continue

        if node.op_type == "Max":
            a = get(node.input[0])
            b = get(node.input[1])
            hc_max = HCMax(operands=[a, b], result_types=[i32])
            ops.append(hc_max)
            env[node.output[0]] = hc_max.results[0]
            continue

        if node.op_type == "Min":
            a = get(node.input[0])
            b = get(node.input[1])
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

    fn = func.FuncOp(fn_name, ([i32], [i32]))
    fn.body = Region(entry_block)

    return ModuleOp(ops=[fn])
