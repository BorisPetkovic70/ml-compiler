"""Loop interchange: swaps the two innermost loops of a matmul nest.

Lowered `hc.matmul` is `i, j, k` loops around
`C[i, j] = C[i, j] + A[i, k] * B[k, j]`. With `k` innermost, each step reads
`B` one row further down. After the swap the order is `i, k, j`: the innermost
loop walks `C` and `B` along a row, and `A[i, k]` does not change in it.

The pass runs on bufferized IR. It reorders a nest whose innermost body
updates a buffer in place, which is how it tells a matmul nest from an
element-wise one. Every other nest is left unchanged.
"""
from xdsl.dialects import func, memref, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block


def apply_loop_interchange(module: ModuleOp) -> None:
    """Swaps the two innermost loops of every matmul nest in `module`, in
    place."""
    for fn in module.ops:
        if isinstance(fn, func.FuncOp):
            for op in list(fn.body.block.ops):
                if isinstance(op, scf.ForOp):
                    _interchange(op)


def _interchange(loop: scf.ForOp) -> None:
    """Swaps the two innermost loops of the nest that starts at `loop`.

    The inner loop moves to where the outer one stood, and the outer loop
    moves inside it, around the same body. Each loop keeps its own bounds and
    induction variable, so the body's ops are not changed.
    """
    outer = None
    while isinstance(loop.body.block.first_op, scf.ForOp):
        outer, loop = loop, loop.body.block.first_op
    inner = loop
    if outer is None or not _updates_in_place(inner.body.block):
        return

    work = list(inner.body.block.ops)[:-1]  # every op but the closing scf.yield
    for op in work:
        op.detach()
    inner.detach()
    outer.parent.insert_op_before(inner, outer)
    outer.detach()
    inner.body.block.insert_op_before(outer, inner.body.block.last_op)
    for op in work:
        outer.body.block.insert_op_before(op, outer.body.block.last_op)


def _updates_in_place(body: Block) -> bool:
    """True if `body` loads an element of a buffer and stores to that same
    element, as `C[i, j] = C[i, j] + ...` does."""
    loads = [op for op in body.ops if isinstance(op, memref.LoadOp)]
    stores = [op for op in body.ops if isinstance(op, memref.StoreOp)]
    return any(
        load.memref is store.memref and list(load.indices) == list(store.indices)
        for load in loads for store in stores
    )
