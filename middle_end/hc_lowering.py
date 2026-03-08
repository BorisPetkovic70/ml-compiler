from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Operation, Region
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

        if op.name not in (
            "hc.add", "hc.mul", "hc.sub", "hc.relu", "hc.pow", "hc.max", "hc.min"
        ):
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
        # ---------------- hc.pow ----------------
        if op.name == "hc.pow":
            base, exp = op.operands
            ty = op.results[0].type  # integer type

            # constants for loop bounds/step
            c0 = arith.ConstantOp.from_int_and_width(0, 32)
            c1 = arith.ConstantOp.from_int_and_width(1, 32)

            # init accumulator = 1
            init = arith.ConstantOp.from_int_and_width(1, 32)

            # scf.for (%i = 0; %i < exp; %i += 1) iter_args(%acc = init) -> (ty) { %acc2 = muli %acc, base; scf.yield %acc2 }
            body = Block(arg_types=[ty, ty])

            # Better: use index for iv and keep acc as ty.
            # Let's construct it robustly:
            iv_ty = builtin.IndexType()
            body = Block(arg_types=[iv_ty, ty])
            iv, acc = body.args

            mul = arith.MuliOp(acc, base)
            body.add_ops([mul, scf.YieldOp(mul.result)])

            loop = scf.ForOp(
                lb=c0.result,          # lower bound
                ub=exp,                # upper bound (dynamic)
                step=c1.result,        # step
                iter_args=[init.result],
                body=Region(body),
            )

            # scf.ForOp returns the iter_args results
            rewriter.replace_op(
                op,
                new_ops=[c0, c1, init, loop],
                new_results=[loop.results[0]],
                safe_erase=True,
            )
            return
        # ---------------- hc.max ----------------
        if op.name == "hc.max":
            lhs, rhs = op.operands
            ty = op.results[0].type

            cmp = arith.CmpiOp(lhs, rhs, "sgt")  # signed greater-than

            then_block = Block(arg_types=[])
            then_block.add_ops([scf.YieldOp(lhs)])

            else_block = Block(arg_types=[])
            else_block.add_ops([scf.YieldOp(rhs)])

            if_op = scf.IfOp(
                cmp.result,
                [ty],
                Region(then_block),
                Region(else_block),
            )

            rewriter.replace_op(
                op,
                new_ops=[cmp, if_op],
                new_results=[if_op.results[0]],
                safe_erase=True,
            )
            return
        if op.name == "hc.min":
            lhs, rhs = op.operands
            ty = op.results[0].type

            cmp = arith.CmpiOp(lhs, rhs, "slt")  # signed less-than

            then_block = Block(arg_types=[])
            then_block.add_ops([scf.YieldOp(lhs)])

            else_block = Block(arg_types=[])
            else_block.add_ops([scf.YieldOp(rhs)])

            if_op = scf.IfOp(
                cmp.result,
                [ty],
                Region(then_block),
                Region(else_block),
            )

            rewriter.replace_op(
                op,
                new_ops=[cmp, if_op],
                new_results=[if_op.results[0]],
                safe_erase=True,
            )
            return



