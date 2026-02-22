from xdsl.dialects import func
from xdsl.ir import Block, Operation, SSAValue
# -----------------------------------------------------------------------------
# Dead code elimination
# -----------------------------------------------------------------------------
def apply_dce(module) -> None:
    """
    Repeatedly run DCE on all function body blocks until fixpoint.
    """
    changed = True
    while changed:
        changed = False
        for top_block in module.body.blocks:
            for op in list(top_block.ops):
                if isinstance(op, func.FuncOp):
                    for body_block in op.body.blocks:
                        if dce_block(body_block):
                            changed = True


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _num_uses(v: SSAValue) -> int:
    uses = getattr(v, "uses", None)
    if uses is None:
        return 0
    try:
        return len(uses)
    except TypeError:
        # Sometimes uses is an iterator/generator-like
        return sum(1 for _ in uses)


def _is_trivially_dceable(op: Operation) -> bool:
    """
    Conservative: only remove arith ops (constants and arithmetic) that have
    no users. Extend later if you add memref/loads/stores/calls etc.
    """
    if op.name == "func.return":
        return False

    # Only drop ops from these dialects for now
    if op.name.startswith("arith."):
        return True

    # If we also want to also drop leftover hc.* after lowering, we could include:
    # if op.name.startswith("hc."):
    #     return True

    return False


def dce_block(block: Block) -> bool:
    """
    One pass of DCE over a block.
    Returns True if anything was erased.
    """
    changed = False

    # Iterate backwards: safer when erasing
    for op in list(block.ops)[::-1]:
        # Must be safe to erase
        if not _is_trivially_dceable(op):
            continue

        # If op has no results, keep it (could be side-effecting; conservative)
        if len(op.results) == 0:
            continue

        # If any result is used, keep it
        if any(_num_uses(res) > 0 for res in op.results):
            continue

        # Otherwise erase
        _erase_op(op)
        changed = True

    return changed



def _erase_op(op: Operation) -> None:
    # Detach from parent block first (required in our xdsl)
    if hasattr(op, "detach"):
        op.detach()
    else:
        raise RuntimeError("Operation.detach() not found; cannot safely DCE ops.")
    op.erase()

