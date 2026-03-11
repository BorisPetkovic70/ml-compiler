from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects import func, arith

from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu, HCPow, HCMax, HCMin
from front_end.build_model import MODEL_PATH
from front_end.loader import import_onnx_to_hc_module

from middle_end.pipeline import (
    MiddleEndPipeline, MiddleEndPipelineConfig
)
from simulator.interpreter import Interpreter

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
    ctx = build_context()
    module = import_onnx_to_hc_module(ctx, MODEL_PATH, "my_func")

    print("=== HIGH LEVEL (hc.*) ===")
    print(module)
    input_val = [31]

    hc_interpreter = Interpreter()
    result_before = hc_interpreter.run_module(module, args=input_val, func_name="my_func")
    print("\nResult (interpreted, before lowering):", result_before)

    config = MiddleEndPipelineConfig(
        apply_lowering=True,
        apply_constant_folding=False,
        apply_dce=False,
        run_analysis=False,
        debug_mode=False,
    )

    middle_end_pipeline = MiddleEndPipeline(config)
    middle_end_pipeline.apply_passes(module)
    print("=== After lowering ===")
    print(module)

    lower_interpreter = Interpreter()
    result_after = lower_interpreter.run_module(module, args=input_val, func_name="my_func")
    print("\nResult (interpreted, after lowering):", result_after)


if __name__ == "__main__":
    main()
