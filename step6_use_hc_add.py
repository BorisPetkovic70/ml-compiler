from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects import func, arith

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


def main():
    ctx = build_context()
    module = build_module(ctx)
    print(module)


if __name__ == "__main__":
    main()
