"""Loop tiling: runs a matmul nest one `T x T` block at a time.

Each of the two loops around the innermost one is split in two: a tile loop
that steps `T` at a time, and a point loop that walks one tile. For the
`i, k, j` order with tile size `T`:

    for ii = 0 to M step T
      for kk = 0 to K step T
        for i = ii to min(ii + T, M)
          for k = kk to min(kk + T, K)
            for j = 0 to N
              C[i, j] = C[i, j] + A[i, k] * B[k, j]

The rows of `B` that one tile reads are reused for every `i` in the tile, so
they stay in the cache. The `min` ends the last tile at the loop's bound when
`T` does not divide it. The innermost loop and any batch loops are left
unchanged.

The pass runs on bufferized IR. It tiles a nest whose innermost body updates
a buffer in place, the same check loop interchange uses. Every other nest is
left unchanged. It assumes a perfect nest whose bounds are defined outside
it, which is what the lowering emits.
"""
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block

from .loop_interchange import _updates_in_place


def apply_loop_tiling(module: ModuleOp, tile_size: int) -> None:
    """Tiles the two loops around the innermost loop of every matmul nest in
    `module`, in place."""
    for fn in module.ops:
        if isinstance(fn, func.FuncOp):
            for op in list(fn.body.block.ops):
                if isinstance(op, scf.ForOp):
                    _tile(op, tile_size)


def _tile(nest: scf.ForOp, tile_size: int) -> None:
    """Tiles the nest that starts at `nest`.

    The two loops being tiled become the point loops: each keeps its induction
    variable and gets new bounds, so the body's ops are not changed. The new
    tile loops take over their old bounds and go around them.
    """
    loops = [nest]
    while isinstance(loops[-1].body.block.first_op, scf.ForOp):
        loops.append(loops[-1].body.block.first_op)
    if not _updates_in_place(loops[-1].body.block):
        return

    size = arith.ConstantOp.from_int_and_width(tile_size, IndexType())
    nest.parent.insert_op_before(size, nest)

    first = loops[-3]
    block, before = first.parent, first  # where the next tile loop goes
    bounds = []
    for point in loops[-3:-1]:
        body = Block([scf.YieldOp()], arg_types=[IndexType()])
        start = body.args[0]
        block.insert_op_before(scf.ForOp(point.lb, point.ub, size, [], body), before)
        block, before = body, body.last_op
        end = arith.AddiOp(start, size)
        stop = arith.MinSIOp(end, point.ub)
        bounds += [end, stop]
        point.operands[0], point.operands[1] = start, stop.result
    first.detach()
    block.insert_ops_before([*bounds, first], before)


""" 
for i_tile = lb_i to ub_i step T:
  for k_tile = lb_k to ub_k step T:
    end_i = i_tile + T
    stop_i = min(end_i, ub_i)
    end_k = k_tile + T
    stop_k = min(end_k, ub_k)
    for i = i_tile to stop_i:
      for k = k_tile to stop_k:
        for j = ...:
          C[i, j] = C[i, j] + A[i, k] * B[k, j]
"""