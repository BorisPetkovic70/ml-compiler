import onnx
from xdsl.dialects import func, arith, builtin

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Region, Block

from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu, HCPow, HCMax, HCMin
from hc_dialect import HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec
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

    #fn = func.FuncOp(fn_name, ([i32], [i32]))
    fn = func.FuncOp(fn_name, ([i32], [i32]), region=Region(entry_block))

    #fn.body = Region(entry_block)
    #fn.body.blocks.append(entry_block)

    return ModuleOp(ops=[fn])


# =============================================================================
#  Vector-aware loader (parallel to the scalar one above)
#  Mirrors the structure of import_onnx_to_hc_module, but binds each ONNX graph
#  input as a vector-typed block argument and maps ONNX ops to the hc.*_vec ops.
#  Pairs with front_end/build_model.build_vec_affine_relu_model.
# =============================================================================

def _vec_type_from_shape(shape) -> builtin.VectorType:
    """Build a VectorType<...xi32> from a list of dimension sizes."""
    dims = [int(d) for d in shape]
    try:
        return builtin.VectorType(dims, i32)
    except TypeError:
        # some xdsl versions take (element_type, shape)
        return builtin.VectorType(i32, dims)


def _vec_type_from_value_info(value_info) -> builtin.VectorType:
    """Derive a VectorType from an ONNX value_info's declared shape."""
    dims = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
    return _vec_type_from_shape(dims)


def _make_vec_const_from_tensor(tensor_proto) -> arith.ConstantOp:
    """Build an arith.constant dense<...> vector from an ONNX initializer tensor."""
    arr = onnx.numpy_helper.to_array(tensor_proto)
    vec_ty = _vec_type_from_shape(arr.shape)
    values = [int(v) for v in arr.reshape(-1).tolist()]
    dense = builtin.DenseIntOrFPElementsAttr.from_list(vec_ty, values)
    return arith.ConstantOp(dense)


def import_onnx_vec_to_hc_module(
    ctx: Context, onnx_path: str, fn_name: str = "main"
) -> ModuleOp:
    model = onnx.load(onnx_path)
    graph = model.graph

    # Map ONNX value names -> SSAValue (results)
    env: dict[str, object] = {}

    # One vector-typed block argument per ONNX graph input.
    input_types = [_vec_type_from_value_info(i) for i in graph.input]
    entry_block = Block(arg_types=input_types)
    for value_info, arg in zip(graph.input, entry_block.args):
        env[value_info.name] = arg

    ops = []

    # 1) Initializers -> dense vector constants
    for init in graph.initializer:
        c = _make_vec_const_from_tensor(init)
        ops.append(c)
        env[init.name] = c.result

    # 2) Node lowering
    for node in graph.node:
        def get(name: str):
            if name not in env:
                raise KeyError(f"ONNX value not found yet: {name} (node {node.op_type})")
            return env[name]

        if node.op_type == "Add":
            a = get(node.input[0])
            b = get(node.input[1])
            hc = HCAddVec(operands=[a, b], result_types=[a.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Sub":
            a = get(node.input[0])
            b = get(node.input[1])
            hc = HCSubVec(operands=[a, b], result_types=[a.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Mul":
            a = get(node.input[0])
            b = get(node.input[1])
            # hc.mul_vec is scalar * vector: detect which operand is the scalar.
            a_is_vec = isinstance(a.type, builtin.VectorType)
            b_is_vec = isinstance(b.type, builtin.VectorType)
            if a_is_vec and b_is_vec:
                hc = HCMulVecVec(operands=[a, b], result_types=[a.type])
                ops.append(hc)
                env[node.output[0]] = hc.results[0]
                continue

            if a_is_vec and not b_is_vec:
                scalar, vec = b, a
            elif b_is_vec and not a_is_vec:
                scalar, vec = a, b
            else:
                raise NotImplementedError("Mul with two scalar operands not supported by the vec loader")
            hc = HCMulVec(operands=[scalar, vec], result_types=[vec.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        if node.op_type == "Relu":
            x = get(node.input[0])
            hc = HCReluVec(operands=[x], result_types=[x.type])
            ops.append(hc)
            env[node.output[0]] = hc.results[0]
            continue

        raise NotImplementedError(f"Unsupported ONNX op (vec loader): {node.op_type}")

    # 3) Output + return
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
