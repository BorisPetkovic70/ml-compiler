import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.dialects import func, arith

from hc_dialect import HiCompiler
from front_end.build_model import build_score_model, MODEL_PATH
from front_end.loader import import_onnx_to_hc_module

from middle_end.pipeline import (
    MiddleEndPipeline, MiddleEndPipelineConfig
)

def build_context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(HiCompiler)
    return ctx

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
            build_score_model(str(model_path))

    original_file = build_dir / f"{model_name}_original.mlir"
    lowered_file = build_dir / f"{model_name}_lowered.mlir"

    ctx = build_context()
    module = import_onnx_to_hc_module(
        ctx, onnx_path=str(model_path), fn_name="my_func"
    )

    print("Saving high-level module...")
    with open(original_file, "w", encoding="utf-8") as f:
        f.write(str(module))

    config = MiddleEndPipelineConfig(
        apply_lowering=True,
        apply_constant_folding=False,
        apply_dce=False,
        run_analysis=False,
        debug_mode=False,
    )

    middle_end_pipeline = MiddleEndPipeline(config)
    middle_end_pipeline.apply_passes(module)

    print("Saving lowered module...")
    with open(lowered_file, "w", encoding="utf-8") as f:
        f.write(str(module))

if __name__ == "__main__":
    main()
