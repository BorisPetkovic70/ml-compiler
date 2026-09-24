import copy
import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, MemRefType, TensorType, VectorType
from xdsl.dialects import func, arith

from hc_dialect import HiCompiler
from front_end.build_model import (
    build_score_model, MODEL_PATH,
    build_vec_affine_relu_model, VEC_AFFINE_RELU_MODEL_PATH,
    build_matmul_model, MATMUL_MODEL_PATH,
)
from front_end.loader import import_onnx_to_hc_module

from middle_end.pipeline import (
    MiddleEndPipeline, MiddleEndPipelineConfig
)
from simulator.interpreter import Interpreter
from back_end.harness_gen import write_harness

def build_context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(HiCompiler)
    return ctx


def _dims_of(ty) -> list[int]:
    return [int(getattr(d, "data", d)) for d in ty.shape]


def _nest(flat: list, dims: list[int]) -> list:
    """Row-major flat list -> nested lists, matching the interpreter's tensor/memref
    representation."""
    if len(dims) <= 1:
        return list(flat)
    step = len(flat) // dims[0]
    return [_nest(flat[r * step:(r + 1) * step], dims[1:]) for r in range(dims[0])]


def _sample_args_for_entry(module, func_name):
    """Auto-generate sample inputs matching each block arg's type: an int for scalar
    (i32) args, a flat list for vector args, a nested list for tensor/memref args.

    Shaped args use the same 10*i + k + 1 formula the generated C harness fills its
    buffers with, so the interpreted result printed here should match what the
    compiled binary prints."""
    for top_block in module.body.blocks:
        for op in top_block.ops:
            if isinstance(op, func.FuncOp) and op.sym_name.data == func_name:
                block = list(op.body.blocks)[0]
                args = []
                for i, a in enumerate(block.args):
                    if isinstance(a.type, (VectorType, TensorType, MemRefType)):
                        dims = _dims_of(a.type)
                        n = 1
                        for d in dims:
                            n *= d
                        flat = [10 * i + k + 1 for k in range(n)]
                        args.append(_nest(flat, dims) if len(dims) > 1 else flat)
                    else:
                        args.append(31 + i)
                print(args)
                return args
    raise RuntimeError(f"Function not found: {func_name}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Optional argument
    model_arg = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    model_path = Path(model_arg)

    # Extract base name (without extension)
    model_name = model_path.stem

    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    if not model_path.exists():
        candidate = build_dir / model_path.name
        if candidate.exists():
            model_path = candidate
        else:
            model_path = build_dir / model_path.name
            print(f"Model not found, building: {model_path}")
            # Compare file names, not whole paths: the *_MODEL_PATH constants are full
            # paths (build/<name>.onnx), so comparing them to model_path.name never
            # matched and every unknown model silently fell back to the score model.
            builders = {
                Path(VEC_AFFINE_RELU_MODEL_PATH).name: build_vec_affine_relu_model,
                Path(MATMUL_MODEL_PATH).name: build_matmul_model,
            }
            builders.get(model_path.name, build_score_model)(str(model_path))

    original_file = build_dir / f"{model_name}_original.mlir"
    lowered_file = build_dir / f"{model_name}_lowered.mlir"

    ctx = build_context()
    module = import_onnx_to_hc_module(
        ctx, onnx_path=str(model_path), fn_name="my_func"
    )

    print("=== HIGH LEVEL (hc.*) ===")
    print(module)

    print("Saving high-level module...")
    with open(original_file, "w", encoding="utf-8") as f:
        f.write(str(module))

    sample_args = _sample_args_for_entry(module, "my_func")
    try:
        before = Interpreter().run_module(module, args=sample_args, func_name="my_func")
        print("\nResult (interpreted, before lowering):", before)
    except RuntimeError as e:
        print(f"\n(interpreter skipped before lowering: {e})")
        before = None

    config = MiddleEndPipelineConfig(
        apply_lowering=True,
        apply_constant_folding=False,
        apply_dce=False,
        run_analysis=False,
        debug_mode=True,
    )

    middle_end_pipeline = MiddleEndPipeline(config)
    middle_end_pipeline.apply_passes(module)

    print("\n=== AFTER LOWERING (arith.*) ===")
    print(module)

    print("Saving lowered module...")
    with open(lowered_file, "w", encoding="utf-8") as f:
        f.write(str(module))

    # Generated *after* the middle end, not before it: bufferization rewrites tensor
    # arguments/results to memref, which changes the entry signature the harness has
    # to match (for scalar/vector models the signature is unchanged either way).
    back_end_dir = Path("back_end")
    harness_file = back_end_dir / f"{model_name}_harness.c"
    write_harness(module, harness_file, func_name="my_func")
    print(f"Wrote harness: {harness_file}")

    try:
        # deepcopy: memref.store mutates its buffer in place, so the post-lowering run
        # must not see buffers the pre-lowering run may have written through.
        after = Interpreter().run_module(
            module, args=copy.deepcopy(sample_args), func_name="my_func"
        )
        print("\nResult (interpreted, after lowering):", after)
    except RuntimeError as e:
        print(f"\n(interpreter skipped after lowering: {e})")
        after = None

    if before is not None and after is not None:
        assert before == after, f"MISMATCH: before={before} after={after}"
        print("OK: interpreter result unchanged by lowering")

if __name__ == "__main__":
    main()
