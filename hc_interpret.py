"""Interpreter driver: `python hc_interpret.py [model.onnx] [args...]`, run
from the repository root.

Runs the model in the Python interpreter before and after the middle end, on
the same arguments the compiled executable takes, and prints the result in the
generated C harness's output format. Exits with an error if the two runs
disagree.

Arguments are read like the harness reads them (back_end/harness_gen.py): one
integer per scalar and per element, in argument order. Any value not supplied
defaults to `10 * (i + 1)` for scalar argument i, or `10*i + k + 1` for
element k of shaped argument i.

Only the result is written to stdout; everything else goes to stderr.
"""
import contextlib
import copy
import sys

from xdsl.dialects import func
from xdsl.dialects.builtin import MemRefType, TensorType

from front_end.build_model import MODEL_PATH
from hc_main import load_module, resolve_model_path, run_middle_end
from simulator.interpreter import Interpreter


def _entry(module, func_name: str) -> func.FuncOp:
    for op in module.ops:
        if isinstance(op, func.FuncOp) and op.sym_name.data == func_name:
            return op
    raise RuntimeError(f"Function not found: {func_name}")


def _nest(flat: list, dims: list[int]) -> list:
    """Row-major flat list -> nested lists of shape `dims`."""
    if len(dims) <= 1:
        return list(flat)
    step = len(flat) // dims[0]
    return [_nest(flat[r * step:(r + 1) * step], dims[1:]) for r in range(dims[0])]


def args_from_argv(fn: func.FuncOp, values: list[int]) -> list:
    """Returns interpreter arguments for `fn`, taking one of `values` per
    scalar and per element in argument order, then the harness defaults.
    Raises ValueError if `values` holds more than `fn` takes."""
    remaining = list(values)

    def take(default: int) -> int:
        return remaining.pop(0) if remaining else default

    args = []
    for i, ty in enumerate(fn.function_type.inputs):
        if isinstance(ty, (TensorType, MemRefType)):
            dims = [d.data for d in ty.shape]
            count = 1
            for d in dims:
                count *= d
            flat = [take(10 * i + k + 1) for k in range(count)]
            args.append(_nest(flat, dims))
        else:
            args.append(take(10 * (i + 1)))

    if remaining:
        raise ValueError(
            f"{len(values)} argument values given, but the model takes only "
            f"{len(values) - len(remaining)}"
        )
    return args


def format_result(value) -> str:
    """Formats an interpreter result like the harness prints it: a scalar as
    a number, and a shaped value as one `[a, b, ...]` line per innermost row."""
    if not isinstance(value, list):
        return str(value)
    if value and isinstance(value[0], list):
        return "\n".join(format_result(row) for row in value)
    return "[" + ", ".join(str(v) for v in value) + "]"


def main():
    # stdout carries only the result, so the model builders' messages go to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        model_path = resolve_model_path(sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH)
    module = load_module(model_path)
    args = args_from_argv(_entry(module, "my_func"), [int(v) for v in sys.argv[2:]])

    # Each run gets its own copy: memref.store mutates buffers in place.
    before = Interpreter().run_module(module, args=copy.deepcopy(args), func_name="my_func")
    run_middle_end(module)
    after = Interpreter().run_module(module, args=copy.deepcopy(args), func_name="my_func")

    if before != after:
        sys.exit(
            "MISMATCH: the interpreter result changed in the middle end\n"
            f"before:\n{format_result(before)}\nafter:\n{format_result(after)}"
        )
    print(format_result(after))


if __name__ == "__main__":
    main()
