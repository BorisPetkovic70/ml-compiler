from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects import func, arith


def build_context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    return ctx


def build_module_with_body(ctx: Context) -> ModuleOp:
    # Create constants: 1 and 2 as i32
    c1 = arith.ConstantOp.from_int_and_width(1, 32)
    c2 = arith.ConstantOp.from_int_and_width(2, 32)

    # Add them: %2 = arith.addi %0, %1 : i32
    add = arith.AddiOp(c1.result, c2.result)

    # Return the result
    ret = func.ReturnOp(add.result)

    # Build a block that contains all operations in order
    body_block = Block(ops=[c1, c2, add, ret])

    # Create a function @my_func() -> i32 with that body
    fn = func.FuncOp(
        "my_func",
        ([], [i32]),        # no arguments, one i32 result
    )
    fn.body = Region(body_block)

    # Put the function inside a module
    module = ModuleOp(ops=[fn])
    return module


def main() -> None:
    ctx = build_context()
    module = build_module_with_body(ctx)
    print(module)


if __name__ == "__main__":
    main()
