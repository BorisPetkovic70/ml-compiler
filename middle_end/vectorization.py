"""Vectorization: runs an innermost loop `W` elements at a time.

The loop must store along a row: its last op stores to `R[..., j]`, where `j`
is its induction variable. Then `W` steps of `j` become one step on
`vector<W x i32>` values. For the `i, k, j` matmul:

    for j = 0 to N - N % W step W
      C[i, j .. j+W] = C[i, j .. j+W] + broadcast(A[i, k]) * B[k, j .. j+W]
    for j = N - N % W to N
      C[i, j] = C[i, j] + A[i, k] * B[k, j]

A load or store indexed by `j` becomes a `vector.load` or `vector.store`, and
the `arith` ops run on vectors. A scalar, such as `A[i, k]`, a constant or a
value from outside the loop, is copied into every lane by `vector.broadcast`
where an `arith` op or the store uses it. The second loop is the original
one, and it runs the elements left over when `W` does not divide `N`.

The element-wise nests of `hc.add`, `hc.sub`, `hc.mul` and `hc.relu` and the
zero-fill nest of matmul store along a row, so they are vectorized too.

Two kinds of loop are left unchanged:
- In the `i, j, k` matmul the innermost loop is `k`, and it stores to the
  single element `C[i, j]` on every step. There is no row to store a vector
  to, so matmul vectorizes only after loop interchange.
- A body with an op other than a load, a constant or one of `_ARITH_OPS`
  before its store, such as the `scf.if` of `hc.max` and `hc.min`.

The pass runs on bufferized IR. It assumes what the lowering emits: the loop
starts at 0 with a constant upper bound, and the buffer it stores to is not
one it also reads at a different element.
"""
from xdsl.dialects import arith, memref, scf, vector
from xdsl.dialects.builtin import IndexType, ModuleOp, VectorType
from xdsl.ir import Block, SSAValue

_ARITH_OPS = (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.MaxSIOp, arith.MinSIOp)


def apply_vectorization(module: ModuleOp, width: int) -> None:
    """Vectorizes every loop in `module` that stores along a row, in place,
    with `width` elements per vector."""
    for op in list(module.walk()):
        if isinstance(op, scf.ForOp) and _stores_along_a_row(op):
            _vectorize(op, width)


def _stores_along_a_row(loop: scf.ForOp) -> bool:
    """True if `loop`'s body ends by storing to `R[..., j]`, with `j` its
    induction variable, and every op before the store is one `_vectorize`
    can rewrite."""
    *work, last = list(loop.body.block.ops)[:-1] or [None]
    return (
        isinstance(last, memref.StoreOp)
        and last.indices[-1] is loop.body.block.args[0]
        and all(isinstance(op, (memref.LoadOp, arith.ConstantOp, *_ARITH_OPS)) for op in work)
    )


def _vectorize(loop: scf.ForOp, width: int) -> None:
    """Puts a vector loop in front of `loop` and leaves `loop` the elements
    the vector loop does not reach.

    The vector loop's body is built op by op from `loop`'s body. `new` maps
    each value of the old body to its value in the new one, a vector or a
    scalar.
    """
    iv = loop.body.block.args[0]
    work = list(loop.body.block.ops)[:-1]  # every op but the closing scf.yield
    n = loop.ub.owner.value.value.data
    if n < width:
        return

    vec = VectorType(work[-1].value.type, [width])
    body = Block([scf.YieldOp()], arg_types=[IndexType()])
    new = {iv: body.args[0]}

    def lanes(value: SSAValue) -> SSAValue:
        """The vector for `value`. A scalar is broadcast first."""
        value = new.get(value, value)  # a value from outside the loop is itself
        if isinstance(value.type, VectorType):
            return value
        broadcast = vector.BroadcastOp(value, vec)
        body.insert_op_before(broadcast, body.last_op)
        return broadcast.vector

    for op in work:
        if isinstance(op, arith.ConstantOp):
            new_op = op.clone()
        elif isinstance(op, _ARITH_OPS):
            new_op = type(op)(lanes(op.lhs), lanes(op.rhs))
        else:  # a load or the store
            idx = [new.get(i, i) for i in op.indices]
            if isinstance(op, memref.StoreOp):
                new_op = vector.StoreOp(lanes(op.value), op.memref, idx)
            elif iv in op.indices:
                new_op = vector.LoadOp(op.memref, idx, vec)
            else:  # the same element on every step
                new_op = memref.LoadOp.get(op.memref, idx)
        if new_op.results:
            new[op.results[0]] = new_op.results[0]
        body.insert_op_before(new_op, body.last_op)

    step = arith.ConstantOp.from_int_and_width(width, IndexType())
    split = arith.ConstantOp.from_int_and_width(n - n % width, IndexType())
    loop.parent.insert_ops_before(
        [step, split, scf.ForOp(loop.lb, split, step, [], body)], loop)
    if n % width:
        loop.operands[0] = split.result
    else:
        loop.detach()
        loop.erase()
