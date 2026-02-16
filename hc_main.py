import inspect
from xdsl.context import Context
from xdsl.ir import Region, Block, Operation
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

# -----------------------------------------------------------------------------
#  Lowering pattern: hc.add -> arith.addi
# -----------------------------------------------------------------------------

class LowerHCAddPattern(RewritePattern):
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

def apply_lowering(module: ModuleOp) -> None:
    applier = GreedyRewritePatternApplier([LowerHCAddPattern()])

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
                else:
                    # fallback: rewrite_op on the function op if that implies walking regions in your build
                    walker.rewrite_op(op)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ctx = build_context()
    module = build_module(ctx)

    print("=== BEFORE LOWERING ===")
    print(module)

    apply_lowering(module)

    print("\n=== AFTER LOWERING ===")
    print(module)


if __name__ == "__main__":
    main()
