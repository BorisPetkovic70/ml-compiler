from xdsl.dialects import arith
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter
)

# -----------------------------------------------------------------------------
# Constant folding pattern
# -----------------------------------------------------------------------------
class FoldArithInts(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Special-case: fold relu lowering if it became maxsi
        if op.name == "arith.maxsi":
            args = _get_const_int_binop_args(op)
            if args is None:
                return
            a, b = args
            width = _result_width(op, 32)
            c = arith.ConstantOp.from_int_and_width(max(a, b), width)
            rewriter.replace_op(op, new_ops=[c], new_results=[c.result], safe_erase=True)
            return

        # Fold binary integer ops when both operands are integer constants
        if op.name not in ("arith.addi", "arith.muli", "arith.subi"):
            return

        args = _get_const_int_binop_args(op)
        if args is None:
            return
        a, b = args

        res_by_op = {
            "arith.addi": a + b,
            "arith.muli": a * b,
            "arith.subi": a - b,
        }
        res = res_by_op[op.name]

        width = _result_width(op, 32)
        c = arith.ConstantOp.from_int_and_width(res, width)
        rewriter.replace_op(op, new_ops=[c], new_results=[c.result], safe_erase=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _defining_op(val: SSAValue) -> Operation | None:
    # SSAValue in xdsl typically has .owner (def op)
    return getattr(val, "owner", None)

def _get_const_int_binop_args(op: Operation):
    """Return (a, b) if both operands are integer constants, else None."""
    if len(op.operands) != 2:
        return None

    lhs, rhs = op.operands
    lhs_op = _defining_op(lhs)
    rhs_op = _defining_op(rhs)
    if lhs_op is None or rhs_op is None:
        return None

    a = _int_from_constant_op(lhs_op)
    b = _int_from_constant_op(rhs_op)
    if a is None or b is None:
        return None

    return a, b

def _int_from_constant_op(op: Operation) -> int | None:
    """
    Extract integer from arith.constant.
    Returns None if not an integer constant we can read.
    """
    if op.name != "arith.constant":
        return None

    # In many versions: arith.ConstantOp has attribute `.value`
    v = getattr(op, "value", None)
    if v is None:
        return None

    # Common shapes: IntegerAttr(value=...), or has `.value`/`.data`
    for attr_name in ("value", "data"):
        if hasattr(v, attr_name):
            try:
                return int(getattr(v, attr_name))
            except Exception:
                pass

    # Last resort: try to parse int from string like "1 : i32"
    # (kept minimal; only if the above fails)
    try:
        s = str(op)
        # crude parse: find "arith.constant " then grab next token
        if "arith.constant" in s:
            tok = s.split("arith.constant", 1)[1].strip().split()[0]
            return int(tok)
    except Exception:
        pass

    return None


def _result_width(op: Operation, default: int = 32) -> int:
    # Try to read integer width from result type if available
    if not op.results:
        return default

    ty = op.results[0].type
    w = getattr(ty, "width", None)
    if w is None:
        return default

    # In your xdsl, width is an IntAttr, not a Python int
    for attr_name in ("data", "value"):
        if hasattr(w, attr_name):
            return int(getattr(w, attr_name))

    # fallback: try direct int conversion if supported
    try:
        return int(w)
    except Exception:
        return default
