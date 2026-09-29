"""Constant folding of integer arith ops on scalar and dense-vector constants."""
from xdsl.dialects import arith, builtin
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter
)

# -----------------------------------------------------------------------------
# Constant folding pattern
# -----------------------------------------------------------------------------
class FoldArithInts(RewritePattern):
    """Replaces an `arith.addi`/`subi`/`muli`/`maxsi` whose operands are both
    `arith.constant`s, or a `vector.broadcast` of a scalar constant, with one
    `arith.constant` (a scalar broadcasts against a vector)."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Fold a vector.broadcast of a constant scalar into a dense vector
        # constant, so a downstream arith op on it becomes foldable too.
        if op.name == "vector.broadcast":
            (src,) = op.operands
            val = _value_from_constant_op(_defining_op(src))
            if val is None or isinstance(val, list):
                return
            res_ty = op.results[0].type
            splatted = [val] * _vec_len(res_ty)
            c = _make_const(splatted, res_ty)
            rewriter.replace(op, new_ops=[c], new_results=[c.result], safe_erase=True)
            return

        # Special-case: fold relu lowering if it became maxsi
        if op.name == "arith.maxsi":
            args = _get_const_binop_args(op)
            if args is None:
                return
            a, b = args
            res = _elt_binop(a, b, max)
            c = _make_const(res, op.results[0].type)
            rewriter.replace(op, new_ops=[c], new_results=[c.result], safe_erase=True)
            return

        # Fold binary integer ops when both operands are constants (scalar or vector)
        if op.name not in ("arith.addi", "arith.muli", "arith.subi"):
            return

        args = _get_const_binop_args(op)
        if args is None:
            return
        a, b = args

        fold_fn_by_op = {
            "arith.addi": lambda x, y: x + y,
            "arith.muli": lambda x, y: x * y,
            "arith.subi": lambda x, y: x - y,
        }
        res = _elt_binop(a, b, fold_fn_by_op[op.name])

        c = _make_const(res, op.results[0].type)
        rewriter.replace(op, new_ops=[c], new_results=[c.result], safe_erase=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _defining_op(val: SSAValue) -> Operation | None:
    if isinstance(val, OpResult):
        return val.op
    return None

def _elt_binop(a, b, f):
    """Apply f element-wise, broadcasting a scalar against a vector (list)."""
    if isinstance(a, list) and isinstance(b, list):
        return [f(x, y) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [f(x, b) for x in a]
    if isinstance(b, list):
        return [f(a, y) for y in b]
    return f(a, b)

def _get_const_binop_args(op: Operation):
    """Return (a, b) if both operands are constants, else None. Each of a/b is
    a python int (scalar constant) or list[int] (dense vector constant)."""
    if len(op.operands) != 2:
        return None

    lhs, rhs = op.operands
    a = _value_from_constant_op(_defining_op(lhs))
    b = _value_from_constant_op(_defining_op(rhs))
    if a is None or b is None:
        return None

    return a, b

def _dense_values(attr) -> list[int] | None:
    """DenseIntOrFPElementsAttr -> list[int], else None."""
    if attr is None or not hasattr(attr, "get_values"):
        return None
    try:
        return list(attr.get_values())
    except Exception:
        return None

def _value_from_constant_op(op: Operation | None):
    """
    arith.constant -> python int (scalar) or list[int] (dense vector).
    Returns None if not a constant we can read.
    """
    if op is None:
        return None

    if op.name != "arith.constant":
        return None

    v = getattr(op, "value", None)
    if v is None:
        return None

    dense = _dense_values(v)
    if dense is not None:
        return dense

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


def _result_width(ty, default: int = 32) -> int:
    # Try to read integer width from a scalar integer type
    w = getattr(ty, "width", None)
    if w is None:
        return default

    # IntegerType.width is an IntAttr, not a Python int
    for attr_name in ("data", "value"):
        if hasattr(w, attr_name):
            return int(getattr(w, attr_name))

    # fallback: try direct int conversion if supported
    try:
        return int(w)
    except Exception:
        return default


def _vec_len(vec_type: builtin.VectorType) -> int:
    n = 1
    for d in vec_type.shape:
        n *= int(getattr(d, "data", d))
    return n


def _make_const(value, ty) -> arith.ConstantOp:
    """Build an arith.constant matching `ty`: dense vector, or scalar int."""
    if isinstance(ty, builtin.VectorType):
        dense = builtin.DenseIntOrFPElementsAttr.from_list(ty, value)
        return arith.ConstantOp(dense)
    return arith.ConstantOp.from_int_and_width(value, _result_width(ty))
