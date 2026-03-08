from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects import func, arith

from hc_dialect import HiCompiler, HCAdd, HCSub, HCMul, HCRelu, HCPow, HCMax
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

def build_module(ctx: Context) -> ModuleOp:
    c1 = arith.ConstantOp.from_int_and_width(1, 32)
    c2 = arith.ConstantOp.from_int_and_width(2, 32)
    c3 = arith.ConstantOp.from_int_and_width(1025, 32)
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

    # Operand must be SSA value
    c5 = arith.ConstantOp.from_int_and_width(3, 32)
    hc_pow = HCPow(
        operands=[c2, c4], # Base and exponent
        result_types=[i32],
    )
    hc_max = HCMax(
        operands=[hc_pow.results[0], c3.result],
        result_types=[i32],
    )

    # Return the result
    ret = func.ReturnOp(hc_max.results[0])

    # Block containing the constants + hc ops + return
    block = Block(
        ops=[c1, c2, c3, c4, c5, hc_add, hc_mul, hc_sub, hc_relu, hc_pow, hc_max, ret]
    )

    # Function returning i32
    fn = func.FuncOp("my_func", ([], [i32]))
    fn.body = Region(block)

    # Module
    module = ModuleOp(ops=[fn])
    return module

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ctx = build_context()
    module = build_module(ctx)

    print("=== HIGH LEVEL (hc.*) ===")
    print(module)

    hc_interpreter = Interpreter()
    result_before = hc_interpreter.run_module(module, "my_func")
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

    lower_interpreter = Interpreter()
    result_after = lower_interpreter.run_module(module, "my_func")
    print("\nResult (interpreted, after lowering):", result_after)


if __name__ == "__main__":
    main()
