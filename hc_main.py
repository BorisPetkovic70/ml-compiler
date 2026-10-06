"""Driver: `python hc_main.py [model.onnx]`, run from the repository root.

Loads (or builds) the model, writes `build/<stem>_original.mlir`, runs the
middle end (lowering, bufferization and constant CSE; folding and DCE
disabled), and writes `build/<stem>_lowered.mlir` and
`back_end/<stem>_harness.c`.
"""
import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects import func, arith

from hc_dialect import HiCompiler
from front_end.build_model import (
    build_score_model, MODEL_PATH,
    build_vec_affine_relu_model, VEC_AFFINE_RELU_MODEL_PATH,
    build_matmul_model, MATMUL_MODEL_PATH,
    build_chained_tensor_model, CHAINED_TENSOR_MODEL_PATH,
    build_batched_matmul_model, BATCHED_MATMUL_MODEL_PATH,
    build_dense_layer_model, DENSE_LAYER_MODEL_PATH,
)
from front_end.loader import import_onnx_to_hc_module

from middle_end.pipeline import (
    MiddleEndPipeline, MiddleEndPipelineConfig
)
from back_end.harness_gen import write_harness

def build_context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(HiCompiler)
    return ctx


def resolve_model_path(model_arg: str) -> Path:
    """Returns the path of the model `model_arg` names: the path itself if it
    exists, otherwise the file of that name in `build/`, which is built first
    if it is missing."""
    model_path = Path(model_arg)
    if model_path.exists():
        return model_path

    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    model_path = build_dir / model_path.name
    if not model_path.exists():
        print(f"Model not found, building: {model_path}")
        # Keyed by file name (the *_MODEL_PATH constants are full paths);
        # every builder in front_end/build_model.py is listed, and a name
        # not listed here still falls back to the score model.
        builders = {
            Path(MODEL_PATH).name: build_score_model,
            Path(VEC_AFFINE_RELU_MODEL_PATH).name: build_vec_affine_relu_model,
            Path(MATMUL_MODEL_PATH).name: build_matmul_model,
            Path(CHAINED_TENSOR_MODEL_PATH).name: build_chained_tensor_model,
            Path(BATCHED_MATMUL_MODEL_PATH).name: build_batched_matmul_model,
            Path(DENSE_LAYER_MODEL_PATH).name: build_dense_layer_model,
        }
        builders.get(model_path.name, build_score_model)(str(model_path))
    return model_path


def load_module(model_path: Path) -> ModuleOp:
    """Imports the ONNX model at `model_path` as an `hc` module whose entry
    function is `my_func`."""
    return import_onnx_to_hc_module(
        build_context(), onnx_path=str(model_path), fn_name="my_func"
    )


def run_middle_end(module: ModuleOp, debug_mode: bool = False) -> None:
    """Runs lowering, bufferization and constant CSE on `module`, in place."""
    config = MiddleEndPipelineConfig(
        apply_lowering=True,
        apply_constant_folding=False,
        apply_dce=False,
        run_analysis=False,
        debug_mode=debug_mode,
    )
    MiddleEndPipeline(config).apply_passes(module)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    model_path = resolve_model_path(sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH)
    model_name = model_path.stem

    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    original_file = build_dir / f"{model_name}_original.mlir"
    lowered_file = build_dir / f"{model_name}_lowered.mlir"

    module = load_module(model_path)

    print("=== HIGH LEVEL (hc.*) ===")
    print(module)

    print("Saving high-level module...")
    with open(original_file, "w", encoding="utf-8") as f:
        f.write(str(module))

    run_middle_end(module, debug_mode=True)

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


if __name__ == "__main__":
    main()
