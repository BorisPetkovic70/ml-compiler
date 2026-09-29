"""Dead code elimination for side-effect-free ops with no uses."""
from xdsl.dialects import func
from xdsl.ir import Block, Operation, SSAValue
# -----------------------------------------------------------------------------
# Dead code elimination
# -----------------------------------------------------------------------------
def apply_dce(module) -> None:
    """Erases unused `arith.*` and `vector.broadcast` ops from the top-level
    blocks of every func.func in `module`, repeating until nothing changes.
    Ops nested in loop or `if` bodies are not visited."""
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
    """True for side-effect-free ops DCE may erase: `arith.*` and
    `vector.broadcast`."""
    if op.name == "func.return":
        return False

    # Only drop ops from these dialects for now
    if op.name.startswith("arith.") or op.name == "vector.broadcast":
        return True

    # If we also want to also drop leftover hc.* after lowering, we could include:
    # if op.name.startswith("hc."):
    #     return True

    return False


def dce_block(block: Block) -> bool:
    """Erases erasable ops in `block` whose results are all unused, in one
    backward sweep. Returns True if anything was erased."""
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

