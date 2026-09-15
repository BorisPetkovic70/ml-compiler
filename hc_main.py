import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, VectorType
from xdsl.dialects import func, arith

from hc_dialect import HiCompiler
from front_end.build_model import (
    build_score_model, MODEL_PATH,
    build_vec_affine_relu_model, VEC_AFFINE_RELU_MODEL_PATH,
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


def _sample_args_for_entry(module, func_name):
    """Auto-generate sample inputs matching each block arg's type: an int for
    scalar (i32) args, a list[int] of matching length for vector args."""
    for top_block in module.body.blocks:
        for op in top_block.ops:
            if isinstance(op, func.FuncOp) and op.sym_name.data == func_name:
                block = list(op.body.blocks)[0]
                args = []
                for i, a in enumerate(block.args):
                    if isinstance(a.type, VectorType):
                        n = 1
                        for d in a.type.shape:
                            n *= int(getattr(d, "data", d))
                        args.append([10 * i + k + 1 for k in range(n)])
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
            if model_path.name == VEC_AFFINE_RELU_MODEL_PATH:
                build_vec_affine_relu_model(str(model_path))
            else:
                build_score_model(str(model_path))

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

    back_end_dir = Path("back_end")
    harness_file = back_end_dir / f"{model_name}_harness.c"
    
    write_harness(module, harness_file, func_name="my_func")
    print(f"Wrote harness: {harness_file}")

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

    try:
        after = Interpreter().run_module(module, args=sample_args, func_name="my_func")
        print("\nResult (interpreted, after lowering):", after)
    except RuntimeError as e:
        print(f"\n(interpreter skipped after lowering: {e})")
        after = None

    if before is not None and after is not None:
        assert before == after, f"MISMATCH: before={before} after={after}"
        print("OK: interpreter result unchanged by lowering")

if __name__ == "__main__":
    main()
