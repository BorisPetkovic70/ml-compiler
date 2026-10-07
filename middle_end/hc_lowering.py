"""Lowers hc ops to arith/scf/vector/tensor, with value semantics.

- Scalar and vector ops become one arith op each; `hc.mul_vec` first
  broadcasts its scalar with `vector.broadcast`.
- `hc.pow` becomes an `scf.for` multiply loop.
- `hc.max`/`hc.min` become `arith.cmpi` + `scf.if`.
- Tensor ops become `scf.for` nests over `tensor.extract`/`tensor.insert`
  (docs/DESIGN.md Section 3). The binary element-wise ones broadcast: each
  operand is indexed by the result's induction variables, as far as its own
  shape reaches (`_broadcast_extract`).

`LowerHCPattern` returns without rewriting for any op whose name is not in its
dispatch tuple, and also for a listed name that has no branch. A missing
lowering therefore leaves the hc op in the IR; `tests/test_lowering.py`
asserts that none survive.
"""
from xdsl.dialects import arith, builtin, scf, tensor, vector
from xdsl.ir import Block, Operation, Region
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter
)

def _const_i32(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(value, 32)


def _index_constants(*values: int) -> dict[int, arith.ConstantOp]:
    """Returns one `index` constant per distinct value, keyed by the value, so
    equal loop bounds share a constant."""
    idx_ty = builtin.IndexType()
    return {
        v: arith.ConstantOp.from_int_and_width(v, idx_ty) for v in dict.fromkeys(values)
    }


# -----------------------------------------------------------------------------
#  Vector lowering helpers
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


def _vec_num_elements(vec_type: builtin.VectorType) -> int:
    """Total number of elements in a vector type (product of its shape dims)."""
    shape = getattr(vec_type, "shape", None)
    if shape is None:
        raise RuntimeError(f"Vector type has no 'shape' attribute: {vec_type}")
    n = 1
    for d in shape:
        n *= _as_int(d)
    return n


def _const_zero_like(like_type) -> arith.ConstantOp:
    """Returns a constant 0: a dense vector of `like_type` for a vector type,
    otherwise an i32."""
    if isinstance(like_type, builtin.VectorType):
        count = _vec_num_elements(like_type)
        dense = builtin.DenseIntOrFPElementsAttr.from_list(like_type, [0] * count)
        return arith.ConstantOp(dense)
    # scalar integer fallback
    return _const_i32(0)


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
#  Tensor elementwise lowering helper (add/sub/mul/relu_tensor share this)
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


def _build_tensor_elementwise_nest(operands, res_type: builtin.TensorType, compute):
    """Builds a loop nest, one loop per dim of `res_type`, that writes
    `compute` of each element into a new tensor of `res_type`, and returns
    (new_ops, result_value).

    `operands` is [x] for a unary op or [lhs, rhs] for a binary one. Each is
    read with `_broadcast_extract`, so it may be a scalar or a tensor whose
    shape broadcasts to `res_type`'s.
    `compute(a, b_or_None)` returns (ops, result_value) for one element.
    """
    shape = [_as_int(d) for d in res_type.shape]

    def body(ivs, carried):
        read_ops, values = [], []
        for x in operands:
            ops, value = _broadcast_extract(x, ivs, shape)
            read_ops += ops
            values.append(value)
        a_val = values[0]
        b_val = values[1] if len(operands) == 2 else None
        compute_ops, result_val = compute(a_val, b_val)
        inserted = tensor.InsertOp(result_val, carried, ivs)
        return [*read_ops, *compute_ops, inserted], inserted.result

    return _build_nest(shape, res_type, body)


# -----------------------------------------------------------------------------
#  Lowering pattern: hc -> arith
# -----------------------------------------------------------------------------
class LowerHCPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Debug: uncomment if you want to see traversal
        # print("VISIT:", op.name)

        if op.name not in (
            "hc.add", "hc.mul", "hc.sub", "hc.relu", "hc.pow", "hc.max", "hc.min",
            "hc.add_vec", "hc.sub_vec", "hc.mul_vec", "hc.mul_vec_vec", "hc.relu_vec",
            "hc.matmul", "hc.add_tensor", "hc.sub_tensor", "hc.mul_tensor", "hc.relu_tensor",
        ):
            return

        # ---------------- hc.add ----------------
        if op.name == "hc.add":
            lhs, rhs = op.operands
            new_op = arith.AddiOp(lhs, rhs)
            rewriter.replace(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.mul ----------------
        if op.name == "hc.mul":
            lhs, rhs = op.operands
            new_op = arith.MuliOp(lhs, rhs)
            rewriter.replace(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.sub ----------------
        if op.name == "hc.sub":
            lhs, rhs = op.operands
            new_op = arith.SubiOp(lhs, rhs)
            rewriter.replace(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.relu ----------------
        if op.name == "hc.relu":
            x = op.operands[0]
            c0 = _const_i32(0)

            if hasattr(arith, "MaxSIOp"):
                maxop = arith.MaxSIOp(x, c0.result)
                rewriter.replace(
                    op,
                    new_ops=[c0, maxop],
                    new_results=[maxop.result],
                    safe_erase=True,
                )
                return
            else:
                raise RuntimeError(
                    "Cannot lower hc.relu: arith.MaxSIOp needed."
                )
        # ---------------- hc.pow ----------------
        if op.name == "hc.pow":
            base, exp = op.operands
            ty = op.results[0].type  # integer type

            iv_ty = builtin.IndexType()

            # loop-control constants must be index
            c0 = arith.ConstantOp.from_int_and_width(0, iv_ty)
            c1 = arith.ConstantOp.from_int_and_width(1, iv_ty)

            # exponent is i32 -> cast to index for scf.for upper bound
            exp_idx = arith.IndexCastOp(exp, iv_ty)

            # init accumulator = 1
            init = arith.ConstantOp.from_int_and_width(1, 32)

            # scf.for body arguments: (iv: index, acc: i32)
            body = Block(arg_types=[iv_ty, ty])
            iv, acc = body.args

            mul = arith.MuliOp(acc, base)
            body.add_ops([mul, scf.YieldOp(mul.result)])

            loop = scf.ForOp(
                lb=c0.result,          # lower bound
                ub=exp_idx.result,     # upper bound
                step=c1.result,        # step
                iter_args=[init.result],
                body=Region(body),
            )

            # scf.ForOp returns the iter_args results
            rewriter.replace(
                op,
                new_ops=[c0, c1, exp_idx, init, loop],
                new_results=[loop.results[0]],
                safe_erase=True,
            )
            return
        # ---------------- hc.max ----------------
        if op.name == "hc.max":
            lhs, rhs = op.operands
            ty = op.results[0].type

            cmp = arith.CmpiOp(lhs, rhs, "sgt")  # signed greater-than

            then_block = Block(arg_types=[])
            then_block.add_ops([scf.YieldOp(lhs)])

            else_block = Block(arg_types=[])
            else_block.add_ops([scf.YieldOp(rhs)])

            if_op = scf.IfOp(
                cmp.result,
                [ty],
                Region(then_block),
                Region(else_block),
            )

            rewriter.replace(
                op,
                new_ops=[cmp, if_op],
                new_results=[if_op.results[0]],
                safe_erase=True,
            )
            return
        if op.name == "hc.min":
            lhs, rhs = op.operands
            ty = op.results[0].type

            cmp = arith.CmpiOp(lhs, rhs, "slt")  # signed less-than

            then_block = Block(arg_types=[])
            then_block.add_ops([scf.YieldOp(lhs)])

            else_block = Block(arg_types=[])
            else_block.add_ops([scf.YieldOp(rhs)])

            if_op = scf.IfOp(
                cmp.result,
                [ty],
                Region(then_block),
                Region(else_block),
            )

            rewriter.replace(
                op,
                new_ops=[cmp, if_op],
                new_results=[if_op.results[0]],
                safe_erase=True,
            )
            return

        # ==========================================================
        #  Vector operations
        #  arith.* ops operate element-wise on vectors when the
        #  operand/result types match.
        # ==========================================================

        # ---------------- hc.add_vec ----------------
        if op.name == "hc.add_vec":
            lhs, rhs = op.operands
            new_op = arith.AddiOp(lhs, rhs)
            rewriter.replace(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.sub_vec ----------------
        if op.name == "hc.sub_vec":
            lhs, rhs = op.operands
            new_op = arith.SubiOp(lhs, rhs)
            rewriter.replace(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.mul_vec (scalar * vector -> vector) ----------------
        if op.name == "hc.mul_vec":
            scalar, vec = op.operands

            # arith.muli needs matching types, so broadcast the (possibly runtime)
            # scalar to vec's vector type.
            bcast = vector.BroadcastOp(scalar, vec.type)
            mul = arith.MuliOp(vec, bcast.vector)
            rewriter.replace(
                op,
                new_ops=[bcast, mul],
                new_results=[mul.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.mul_vec_vec (vector * vector -> vector) ----------------
        if op.name == "hc.mul_vec_vec":
            lhs, rhs = op.operands
            new_op = arith.MuliOp(lhs, rhs)
            rewriter.replace(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.relu_vec ----------------
        if op.name == "hc.relu_vec":
            x = op.operands[0]
            c0 = _const_zero_like(x.type)  # dense-zero vector of x's type

            if hasattr(arith, "MaxSIOp"):
                maxop = arith.MaxSIOp(x, c0.result)
                rewriter.replace(
                    op,
                    new_ops=[c0, maxop],
                    new_results=[maxop.result],
                    safe_erase=True,
                )
                return
            else:
                raise RuntimeError("Cannot lower hc.relu_vec: arith.MaxSIOp needed.")

        # ==========================================================
        #  Tensor operations
        # ==========================================================

        # ---------------- hc.matmul (MxK * KxN -> MxN) ----------------
        if op.name == "hc.matmul":
            lhs, rhs = op.operands
            new_ops, result = _build_matmul_nest(lhs, rhs, op.results[0].type)
            rewriter.replace(
                op,
                new_ops=new_ops,
                new_results=[result],
                safe_erase=True,
            )
            return

        # ---------------- hc.add_tensor ----------------
        if op.name == "hc.add_tensor":
            lhs, rhs = op.operands

            def compute(a, b):
                add = arith.AddiOp(a, b)
                return [add], add.result

            new_ops, result = _build_tensor_elementwise_nest([lhs, rhs], op.results[0].type, compute)
            rewriter.replace(op, new_ops=new_ops, new_results=[result], safe_erase=True)
            return

        # ---------------- hc.sub_tensor ----------------
        if op.name == "hc.sub_tensor":
            lhs, rhs = op.operands

            def compute(a, b):
                sub = arith.SubiOp(a, b)
                return [sub], sub.result

            new_ops, result = _build_tensor_elementwise_nest([lhs, rhs], op.results[0].type, compute)
            rewriter.replace(op, new_ops=new_ops, new_results=[result], safe_erase=True)
            return

        # ---------------- hc.mul_tensor ----------------
        if op.name == "hc.mul_tensor":
            lhs, rhs = op.operands

            def compute(a, b):
                mul = arith.MuliOp(a, b)
                return [mul], mul.result

            new_ops, result = _build_tensor_elementwise_nest([lhs, rhs], op.results[0].type, compute)
            rewriter.replace(op, new_ops=new_ops, new_results=[result], safe_erase=True)
            return

        # ---------------- hc.relu_tensor ----------------
        if op.name == "hc.relu_tensor":
            (x,) = op.operands

            def compute(a, _b):
                # Zero of the element type: tensor lowerings don't assume i32.
                zero = arith.ConstantOp.from_int_and_width(0, a.type)
                mx = arith.MaxSIOp(a, zero.result)
                return [zero, mx], mx.result

            new_ops, result = _build_tensor_elementwise_nest([x], op.results[0].type, compute)
            rewriter.replace(op, new_ops=new_ops, new_results=[result], safe_erase=True)
            return



