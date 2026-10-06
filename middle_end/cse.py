"""Common-subexpression elimination for constants.

Every lowering creates its own loop bounds and zeros, so a function ends up
with many identical `arith.constant` ops. `apply_constant_cse` keeps one per
distinct value and type, at the top of the function.
"""
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import ModuleOp


def apply_constant_cse(module: ModuleOp) -> None:
    """Merges identical `arith.constant` ops in every func.func in `module`,
    in place."""
    for op in module.ops:
        if isinstance(op, func.FuncOp):
            _cse_function(op)


def _cse_function(fn: func.FuncOp) -> None:
    """Moves the first constant of each value and type to the top of `fn`'s
    entry block, and replaces every later duplicate with it.

    Moving a constant is always legal: it has no operands and no side
    effects, and a value defined at the top of the entry block is visible in
    every op below it, including inside loop bodies.
    """
    entry = fn.body.blocks[0]
    kept: dict[tuple, arith.ConstantOp] = {}
    last = None  # the constant most recently moved to the top
    for op in list(fn.body.walk()):
        if not isinstance(op, arith.ConstantOp):
            continue
        key = (op.value, op.result.type)
        if key in kept:
            op.result.replace_all_uses_with(kept[key].result)
            op.detach()
            op.erase()
            continue
        op.detach()
        if last is not None:
            entry.insert_op_after(op, last)
        elif entry.first_op is not None:
            entry.insert_op_before(op, entry.first_op)
        else:
            entry.add_op(op)
        kept[key] = op
        last = op
