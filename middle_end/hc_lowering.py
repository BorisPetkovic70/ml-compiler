from xdsl.dialects import arith
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter
)

def _const_i32(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(value, 32)

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
