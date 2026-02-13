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
from hc_dialect import HiCompiler, HCAdd


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

    # Our custom operation: %2 = hc.add %0, %1 : i32
    hc_add = HCAdd(
        operands=[c1.result, c2.result],
        result_types=[i32],
    )

    # Return the result
    ret = func.ReturnOp(hc_add.results[0])  # or hc_add.res

    # Block containing the constants + hc_add + return
    block = Block(ops=[c1, c2, hc_add, ret])

    # Function returning i32
    fn = func.FuncOp("my_func", ([], [i32]))
    fn.body = Region(block)

    # Module
    module = ModuleOp(ops=[fn])
    return module

# -----------------------------------------------------------------------------
#  Lowering pattern: hc.add -> arith.addi
# -----------------------------------------------------------------------------

class LowerHCAddPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Debug: uncomment if you want to see traversal
        print("VISIT:", op.name)

        if op.name != "hc.add":
            return

        # hc.add has two operands
        lhs, rhs = op.operands

        # Replacement op
        new_add = arith.AddiOp(lhs, rhs)

        rewriter.replace_op(
            op,
            new_ops=[new_add],
            new_results=[new_add.result],
            safe_erase=True,
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
