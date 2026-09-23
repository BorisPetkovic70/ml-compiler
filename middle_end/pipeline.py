import inspect
from dataclasses import dataclass
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)

from .analysis import analyze
from .bufferization import apply_bufferization
from .constant_folding import FoldArithInts
from .dead_code_elimination import apply_dce
from .hc_lowering import LowerHCPattern
# -----------------------------
# Configuration object
# -----------------------------
@dataclass(frozen=True)
class MiddleEndPipelineConfig:
    run_analysis: bool = False
    debug_mode: bool = False

    apply_lowering: bool = True
    apply_bufferization: bool = True
    apply_constant_folding: bool = True
    apply_dce: bool = True

# -----------------------------
# Pipeline
# -----------------------------
@dataclass
class MiddleEndPipeline:
    def __init__(self, config: MiddleEndPipelineConfig):
        self.config = config

    def apply_passes(self, module: ModuleOp) -> None:
        if self.config.apply_lowering:
            # Apply lowering
            _apply_pass(module, LowerHCPattern)
            self._print_module(module, "=== AFTER LOWERING  (arith.*) ===")

        if self.config.apply_bufferization:
            # Bufferize any function using tensors (a no-op on scalar/vector-only ones)
            apply_bufferization(module)
            self._print_module(module, "=== AFTER BUFFERIZATION (memref.*) ===")

        if self.config.apply_constant_folding:
            # Apply constant folding
            _apply_pass(module, FoldArithInts)
            self._print_module(module, "=== AFTER CONSTANT FOLDING ===")

        if self.config.apply_dce:
            # Apply dead code elimination
            apply_dce(module)
            self._print_module(module, "=== AFTER DEAD CODE ELIMINATION ===")

        if self.config.run_analysis:
            # Analyze pass
            analyze(module)

    def _print_module(self, module: ModuleOp, message: str):
        if self.config.debug_mode:
            print(message + "\n")
            print(module)


def _apply_pass(module: ModuleOp, pattern: RewritePattern) -> None:
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


