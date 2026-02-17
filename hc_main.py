import inspect
from xdsl.context import Context
from xdsl.ir import Region, Block, Operation, SSAValue
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects import func, arith
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu


def build_context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(HiCompiler)   # <-- our custom dialect
    return ctx

def build_module(ctx: Context) -> ModuleOp:
    # Constants: 1 and 2
    c1 = arith.ConstantOp.from_int_and_width(1, 32)
    c2 = arith.ConstantOp.from_int_and_width(2, 32)
    c3 = arith.ConstantOp.from_int_and_width(3, 32)
    c4 = arith.ConstantOp.from_int_and_width(10, 32)

    hc_add = HCAdd(
        operands=[c1.result, c2.result],
        result_types=[i32],
    )
    hc_mul = HCMul(
        operands=[hc_add.results[0], c3.result],
        result_types=[i32],
    )
    hc_sub = HCSub(
        operands=[hc_mul.results[0], c4.result],
        result_types=[i32],
    )
    hc_relu = HCRelu(
        operands=[hc_sub.results[0]],
        result_types=[i32],
    )

    # Return the result
    ret = func.ReturnOp(hc_relu.results[0])

    # Block containing the constants + hc ops + return
    block = Block(ops=[c1, c2, c3, c4, hc_add, hc_mul, hc_sub, hc_relu, ret])

    # Function returning i32
    fn = func.FuncOp("my_func", ([], [i32]))
    fn.body = Region(block)

    # Module
    module = ModuleOp(ops=[fn])
    return module

def _const_i32(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(value, 32)

def apply_pass(module: ModuleOp, pattern: RewritePattern) -> None:
    applier = GreedyRewritePatternApplier([pattern()])

    # Some xdsl builds have extra kwargs; enable recursion if available.
    init_sig = inspect.signature(PatternRewriteWalker.__init__)
    kwargs = {}
    for k in ("walk_regions", "apply_recursively", "walk_into_regions"):
        if k in init_sig.parameters:
            kwargs[k] = True

    walker = PatternRewriteWalker(applier, **kwargs)

    # Walk/rewrite the function bodies explicitly
    for top_block in module.body.blocks:
        for op in top_block.ops:
            if isinstance(op, func.FuncOp):
                if hasattr(walker, "rewrite_region"):
                    walker.rewrite_region(op.body)
                elif hasattr(walker, "rewrite_op"):
                    walker.rewrite_op(op)
                else:
                    raise RuntimeError("No suitable rewrite entrypoint found on PatternRewriteWalker.")

# -----------------------------------------------------------------------------
#  Lowering pattern: hc -> arith
# -----------------------------------------------------------------------------
class LowerHCPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Debug: uncomment if you want to see traversal
        # print("VISIT:", op.name)

        if op.name not in ("hc.add", "hc.mul", "hc.sub", "hc.relu"):
            return

        # ---------------- hc.add ----------------
        if op.name == "hc.add":
            lhs, rhs = op.operands
            new_op = arith.AddiOp(lhs, rhs)
            rewriter.replace_op(
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
            rewriter.replace_op(
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
            rewriter.replace_op(
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
                rewriter.replace_op(
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
# Dead code elimination
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

def _erase_op(op: Operation) -> None:
    # Detach from parent block first (required in our xdsl)
    if hasattr(op, "detach"):
        op.detach()
    else:
        raise RuntimeError("Operation.detach() not found; cannot safely DCE ops.")
    op.erase()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ctx = build_context()
    module = build_module(ctx)

    print("=== HIGH LEVEL (hc.*) ===")
    print(module)

    # Apply lowering
    apply_pass(module, LowerHCPattern)

    print("\n=== AFTER LOWERING  (arith.*) ===")
    print(module)

    # Apply constant folding
    apply_pass(module, FoldArithInts)
    print("\n=== AFTER CONSTANT FOLDING ===")
    print(module)

    apply_dce(module)
    print("\n=== AFTER DCE ===")
    print(module)


if __name__ == "__main__":
    main()
