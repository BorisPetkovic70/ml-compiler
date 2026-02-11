from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.printer import Printer

from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects import func


def build_context() -> Context:
    ctx = Context()
    # Load the builtin dialect into the context
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    return ctx


def build_module_with_empty_func(ctx: Context) -> ModuleOp:
    # Create an empty module (same idea as step2)
    module = ModuleOp(ops=[])

    # Create an *empty* function: no arguments, no results (for now)
    func_name = "my_empty_func"

    # Most xdsl versions mirror MLIR's `func.func` here:
    #   func.FuncOp(name, (arg_types, result_types))
    # With no args/results: ([], [])
    fn = func.FuncOp(func_name, ([], []))

    # Make the function a declaration by giving it an empty body region.
    # Later, in step4, we'll actually add a block and operations.
    fn.body = Region()

    # Instead of touching module.body.blocks, replace the whole body region
    # with a Region that has a single Block containing `fn`.
    module.body = Region(Block(ops=[fn]))

    return module


def main() -> None:
    ctx = build_context()
    module = build_module_with_empty_func(ctx)

    # The new printer API
    printer = Printer()
    printer.print_op(module)
    print()  # newline

if __name__ == "__main__":
    main()
