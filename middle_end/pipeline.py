"""The middle-end pass pipeline:
lowering -> bufferization -> loop interchange -> loop tiling ->
constant folding -> constant CSE -> DCE.

The order is fixed and load-bearing (docs/DESIGN.md Section 5); each pass is
switched on or off by `MiddleEndPipelineConfig`. Loop interchange and loop
tiling are off by default. `apply_passes` rewrites the module in place and
never calls `module.verify()`: verification is the caller's responsibility.
"""
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
from .cse import apply_constant_cse
from .dead_code_elimination import apply_dce
from .hc_lowering import LowerHCPattern
from .loop_interchange import apply_loop_interchange
from .loop_tiling import apply_loop_tiling
# -----------------------------
# Configuration object
# -----------------------------
@dataclass(frozen=True)
class MiddleEndPipelineConfig:
    """Pass switches. `tile_size` is the tile size for loop tiling, and 0
    turns it off. `debug_mode` prints the module after each pass;
    `run_analysis` prints the analysis.py reports after the last pass."""
    run_analysis: bool = False
    debug_mode: bool = False

    apply_lowering: bool = True
    apply_bufferization: bool = True
    interchange_loops: bool = False
    tile_size: int = 0
    apply_constant_folding: bool = True
    apply_cse: bool = True
    apply_dce: bool = True

# -----------------------------
# Pipeline
# -----------------------------
class MiddleEndPipeline:
    def __init__(self, config: MiddleEndPipelineConfig):
        self.config = config

    def apply_passes(self, module: ModuleOp) -> None:
        """Runs the enabled passes on `module`, in place."""
        if self.config.apply_lowering:
            # Apply lowering
            _apply_pass(module, LowerHCPattern)
            self._print_module(module, "=== AFTER LOWERING (arith.*) ===")

        if self.config.apply_bufferization:
            # Bufferize any function using tensors (a no-op on scalar-only ones)
            apply_bufferization(module)
            self._print_module(module, "=== AFTER BUFFERIZATION (memref.*) ===")

        if self.config.interchange_loops:
            # Reorder each matmul nest from i, j, k to i, k, j
            apply_loop_interchange(module)
            self._print_module(module, "=== AFTER LOOP INTERCHANGE ===")

        if self.config.tile_size:
            # Run each matmul nest one tile at a time
            apply_loop_tiling(module, self.config.tile_size)
            self._print_module(module, "=== AFTER LOOP TILING ===")

        if self.config.apply_constant_folding:
            # Apply constant folding
            _apply_pass(module, FoldArithInts)
            self._print_module(module, "=== AFTER CONSTANT FOLDING ===")

        if self.config.apply_cse:
            # Merge identical constants, including the ones folding created
            apply_constant_cse(module)
            self._print_module(module, "=== AFTER CONSTANT CSE ===")

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
    """Applies `pattern` greedily to the body of every func.func in `module`."""
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


