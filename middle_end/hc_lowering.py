"""Lowers hc ops to arith/scf/tensor, with value semantics.

An element-wise op has a kernel in `_KERNELS`: a function that builds the ops
computing one element from scalar values.

- `hc.add/sub/mul/relu`: one arith op each.
- `hc.pow`: an `scf.for` multiply loop.
- `hc.max`/`hc.min`: `arith.cmpi` + `scf.if`.

An op with a scalar result lowers to its kernel alone. An op with a tensor
result lowers to an `scf.for` nest over `tensor.extract`/`tensor.insert`
with the kernel in the innermost body (docs/DESIGN.md Section 3). The binary
ones broadcast: each operand is indexed by the result's induction variables,
as far as its own shape reaches (`_broadcast_extract`). `hc.matmul` has its
own nest (`_build_matmul_nest`).

`LowerHCPattern` raises NotImplementedError for an `hc` op that has neither
a kernel nor its own nest.
"""
from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.ir import Block, Operation, Region
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter
)

def _index_constants(*values: int) -> dict[int, arith.ConstantOp]:
    """Returns one `index` constant per distinct value, keyed by the value, so
    equal loop bounds share a constant."""
    idx_ty = builtin.IndexType()
    return {
        v: arith.ConstantOp.from_int_and_width(v, idx_ty) for v in dict.fromkeys(values)
    }


# -----------------------------------------------------------------------------
#  Shape helpers
# -----------------------------------------------------------------------------
def _as_int(x) -> int:
    """Convert xDSL int-like objects (IntAttr, nested attrs) to a python int."""
    if isinstance(x, int):
        return x
    if hasattr(x, "data"):
        return int(x.data)
    if hasattr(x, "value") and hasattr(x.value, "data"):
        return int(x.value.data)
    return int(x)


# -----------------------------------------------------------------------------
#  Loop-nest builder
# -----------------------------------------------------------------------------
def _build_nest(shape: list[int], res_type: builtin.TensorType, body_fn):
    """Builds one `scf.for` per entry of `shape`, outermost first, and returns
    (new_ops, result_value).

    Every loop runs from 0 to its dim with step 1 and carries a tensor of
    `res_type` in iter_args. The outermost loop starts from a `tensor.empty`,
    and each inner loop from the tensor its parent carries.

    `body_fn(ivs, carried)` builds the innermost body: `ivs` are the induction
    variables, outermost first, and `carried` is the tensor the innermost loop
    carries. It returns (ops, new_tensor), and `new_tensor` is yielded.
    """
    idx_ty = builtin.IndexType()
    consts = _index_constants(0, 1, *shape)
    c0, c1 = consts[0], consts[1]
    # Its contents are unspecified: the nest is expected to write every element.
    empty_res = tensor.EmptyOp([], res_type)

    # Each body block takes (iv, carried tensor). All are created up front so
    # the innermost body can reference every induction variable.
    bodies = [Block(arg_types=[idx_ty, res_type]) for _ in shape]
    ivs = [body.args[0] for body in bodies]
    ops, new_tensor = body_fn(ivs, bodies[-1].args[1])
    bodies[-1].add_ops([*ops, scf.YieldOp(new_tensor)])

    # Wrap from the inside out: each outer body holds the next loop and yields
    # its result.
    loop = None
    for d in reversed(range(len(shape))):
        if loop is not None:
            bodies[d].add_ops([loop, scf.YieldOp(loop.results[0])])
        init = bodies[d - 1].args[1] if d else empty_res.tensor
        loop = scf.ForOp(c0.result, consts[shape[d]].result, c1.result, [init], Region(bodies[d]))

    return [*consts.values(), empty_res, loop], loop.results[0]


# -----------------------------------------------------------------------------
#  Matmul lowering helper
# -----------------------------------------------------------------------------
def _build_matmul_nest(lhs, rhs, res_type: builtin.TensorType):
    """Builds the loop nest computing `lhs @ rhs` into a new tensor of
    `res_type`, and returns (new_ops, result_value).

    The i and j loops (and the outer batch loop, for rank 3) carry the result
    tensor in iter_args. The innermost k loop carries a scalar accumulator.
    For rank 3, the batch induction variable is prepended to every
    extract/insert index.
    """
    res_dims = [_as_int(d) for d in res_type.shape]
    batched = len(res_dims) == 3
    batch_dim, m, n = (res_dims if batched else (None, *res_dims))
    k_dim = _as_int(list(lhs.type.shape)[-1])
    elem_ty = res_type.element_type
    idx_ty = builtin.IndexType()

    # loop-control constants must be index
    bounds = (m, n, k_dim, batch_dim) if batched else (m, n, k_dim)
    consts = _index_constants(0, 1, *bounds)
    c0, c1, c_m, c_n, c_k = (consts[v] for v in (0, 1, m, n, k_dim))
    # Result tensor (full res_type, batch dim included). Its contents are
    # unspecified: the nest writes every element and reads none.
    empty_res = tensor.EmptyOp([], res_type)
    setup_ops = [*consts.values(), empty_res]

    # The batch loop, when present, is the outermost level: it carries the
    # full result tensor through iter_args exactly like the i loop does below,
    # and its induction variable is prepended to every extract/insert index.
    prefix = []
    i_init = empty_res.tensor
    batch_body = None
    if batched:
        c_bdim = consts[batch_dim]
        batch_body = Block(arg_types=[idx_ty, res_type])
        bi, c_bi = batch_body.args
        prefix = [bi]
        i_init = c_bi

    # Each loop body block takes (iv, iter_arg); create all three up front so the
    # inner bodies can reference i and j.
    i_body = Block(arg_types=[idx_ty, res_type])
    j_body = Block(arg_types=[idx_ty, res_type])
    k_body = Block(arg_types=[idx_ty, elem_ty])
    i, c_i = i_body.args
    j, c_ij = j_body.args
    k, acc = k_body.args

    # k loop: acc += A[i,k] * B[k,j] (A[b,i,k] * B[b,k,j] when batched)
    a_val = tensor.ExtractOp(lhs, [*prefix, i, k], elem_ty)
    b_val = tensor.ExtractOp(rhs, [*prefix, k, j], elem_ty)
    prod = arith.MuliOp(a_val.result, b_val.result)
    new_acc = arith.AddiOp(acc, prod.result)
    k_body.add_ops([a_val, b_val, prod, new_acc, scf.YieldOp(new_acc.result)])

    # j loop: C[i,j] = <k-loop result>, threading the tensor
    zero_acc = arith.ConstantOp.from_int_and_width(0, elem_ty)
    k_loop = scf.ForOp(c0.result, c_k.result, c1.result, [zero_acc.result], Region(k_body))
    inserted = tensor.InsertOp(k_loop.results[0], c_ij, [*prefix, i, j])
    j_body.add_ops([zero_acc, k_loop, inserted, scf.YieldOp(inserted.result)])

    # i loop: yields the tensor produced by the j loop
    j_loop = scf.ForOp(c0.result, c_n.result, c1.result, [c_i], Region(j_body))
    i_body.add_ops([j_loop, scf.YieldOp(j_loop.results[0])])
    i_loop = scf.ForOp(c0.result, c_m.result, c1.result, [i_init], Region(i_body))

    if batched:
        batch_body.add_ops([i_loop, scf.YieldOp(i_loop.results[0])])
        batch_loop = scf.ForOp(
            c0.result, c_bdim.result, c1.result, [empty_res.tensor], Region(batch_body)
        )
        return [*setup_ops, batch_loop], batch_loop.results[0]

    return [*setup_ops, i_loop], i_loop.results[0]


# -----------------------------------------------------------------------------
#  Kernels: one element of an element-wise op
#
#  A kernel takes the op's operands as scalar values and returns
#  (ops, result_value).
# -----------------------------------------------------------------------------
def _add(a, b):
    add = arith.AddiOp(a, b)
    return [add], add.result


def _sub(a, b):
    sub = arith.SubiOp(a, b)
    return [sub], sub.result


def _mul(a, b):
    mul = arith.MuliOp(a, b)
    return [mul], mul.result


def _relu(x):
    """max(x, 0)."""
    zero = arith.ConstantOp.from_int_and_width(0, x.type)
    mx = arith.MaxSIOp(x, zero.result)
    return [zero, mx], mx.result


def _pow(base, exp):
    """A loop that runs `exp` times and multiplies an accumulator, which
    starts at 1, by `base`."""
    idx_ty = builtin.IndexType()
    c0 = arith.ConstantOp.from_int_and_width(0, idx_ty)
    c1 = arith.ConstantOp.from_int_and_width(1, idx_ty)
    # The exponent is an integer; an scf.for bound must be an index.
    exp_idx = arith.IndexCastOp(exp, idx_ty)
    init = arith.ConstantOp.from_int_and_width(1, base.type)

    body = Block(arg_types=[idx_ty, base.type])
    _, acc = body.args
    mul = arith.MuliOp(acc, base)
    body.add_ops([mul, scf.YieldOp(mul.result)])

    loop = scf.ForOp(c0.result, exp_idx.result, c1.result, [init.result], Region(body))
    return [c0, c1, exp_idx, init, loop], loop.results[0]


def _select(predicate: str):
    """Returns a kernel that yields `a` when `a <predicate> b` holds, else
    `b`, through an `scf.if`."""
    def kernel(a, b):
        cmp = arith.CmpiOp(a, b, predicate)
        then_block = Block(arg_types=[])
        then_block.add_ops([scf.YieldOp(a)])
        else_block = Block(arg_types=[])
        else_block.add_ops([scf.YieldOp(b)])
        if_op = scf.IfOp(cmp.result, [a.type], Region(then_block), Region(else_block))
        return [cmp, if_op], if_op.results[0]

    return kernel


_KERNELS = {
    "hc.add": _add,
    "hc.sub": _sub,
    "hc.mul": _mul,
    "hc.relu": _relu,
    "hc.pow": _pow,
    "hc.max": _select("sgt"),  # signed greater-than
    "hc.min": _select("slt"),  # signed less-than
    "hc.add_tensor": _add,
    "hc.sub_tensor": _sub,
    "hc.mul_tensor": _mul,
    "hc.relu_tensor": _relu,
}


# -----------------------------------------------------------------------------
#  Tensor elementwise lowering helper
# -----------------------------------------------------------------------------
def _broadcast_extract(x, ivs, shape: list[int]):
    """Returns (ops, value) for the element of operand `x` that the result
    element at `ivs` is computed from. `shape` is the result's shape.

    A scalar operand is that value itself, with no ops. A tensor operand is
    aligned with the result at the last dim, so it is indexed by the trailing
    induction variables. Where its dim is 1 and the result's is not, the
    index is a constant 0: its one element is reused along that dim.
    """
    if not isinstance(x.type, builtin.TensorType):
        return [], x
    dims = [_as_int(d) for d in x.type.shape]
    res_dims = shape[-len(dims):]
    zero = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
    indices = [
        iv if dim == res_dim else zero.result
        for dim, res_dim, iv in zip(dims, res_dims, ivs[-len(dims):])
    ]
    extract = tensor.ExtractOp(x, indices, x.type.element_type)
    return ([extract] if dims == res_dims else [zero, extract]), extract.result


def _build_tensor_elementwise_nest(operands, res_type: builtin.TensorType, kernel):
    """Builds a loop nest, one loop per dim of `res_type`, that writes
    `kernel` of each element into a new tensor of `res_type`, and returns
    (new_ops, result_value).

    Each operand is read with `_broadcast_extract`, so it may be a scalar or
    a tensor whose shape broadcasts to `res_type`'s.
    """
    shape = [_as_int(d) for d in res_type.shape]

    def body(ivs, carried):
        read_ops, values = [], []
        for x in operands:
            ops, value = _broadcast_extract(x, ivs, shape)
            read_ops += ops
            values.append(value)
        kernel_ops, result_val = kernel(*values)
        inserted = tensor.InsertOp(result_val, carried, ivs)
        return [*read_ops, *kernel_ops, inserted], inserted.result

    return _build_nest(shape, res_type, body)


# -----------------------------------------------------------------------------
#  Lowering pattern: hc -> arith/scf/tensor
# -----------------------------------------------------------------------------
class LowerHCPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not op.name.startswith("hc."):
            return

        res_type = op.results[0].type
        if op.name == "hc.matmul":
            lhs, rhs = op.operands
            new_ops, result = _build_matmul_nest(lhs, rhs, res_type)
        else:
            kernel = _KERNELS.get(op.name)
            if kernel is None:
                raise NotImplementedError(f"No lowering for {op.name}")
            if isinstance(res_type, builtin.TensorType):
                new_ops, result = _build_tensor_elementwise_nest(op.operands, res_type, kernel)
            else:
                new_ops, result = kernel(*op.operands)

        rewriter.replace(op, new_ops=new_ops, new_results=[result], safe_erase=True)
