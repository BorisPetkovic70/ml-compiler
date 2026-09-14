"""Generate a C harness matching an hc module's entry function signature.

Since the compiled pipeline calls the emitted LLVM function directly through a
C `extern` declaration, the harness has to match that function's exact
argument/result types -- which vary per model (scalar-only, vector-only,
multiple inputs, mixed scalar+vector). One static harness.c can't cover all of
them, so this generates a matching one per model instead.
"""
from xdsl.dialects import func
from xdsl.dialects.builtin import VectorType


def _vec_width(ty: VectorType) -> int:
    n = 1
    for d in ty.shape:
        n *= int(getattr(d, "data", d))
    return n


def _type_spec(ty):
    """('scalar', None) or ('vector', width)."""
    if isinstance(ty, VectorType):
        return ("vector", _vec_width(ty))
    return ("scalar", None)


def _c_type_name(spec) -> str:
    kind, n = spec
    return "int" if kind == "scalar" else f"v{n}si"


def _find_entry(module, func_name: str) -> func.FuncOp:
    for top_block in module.body.blocks:
        for op in top_block.ops:
            if isinstance(op, func.FuncOp) and op.sym_name.data == func_name:
                return op
    raise RuntimeError(f"Function not found: {func_name}")


def generate_harness_c(module, func_name: str = "my_func") -> str:
    fn = _find_entry(module, func_name)
    arg_specs = [_type_spec(t) for t in fn.function_type.inputs.data]

    out_types = fn.function_type.outputs.data
    if len(out_types) != 1:
        raise RuntimeError("harness generator expects exactly 1 result")
    ret_spec = _type_spec(out_types[0])

    widths = sorted({n for kind, n in arg_specs + [ret_spec] if kind == "vector"})

    lines = ["#include <stdio.h>", "#include <stdlib.h>", ""]
    for n in widths:
        lines.append(f"typedef int v{n}si __attribute__((vector_size({4 * n})));")
    if widths:
        lines.append("")

    c_arg_types = [_c_type_name(s) for s in arg_specs]
    c_ret_type = _c_type_name(ret_spec)
    lines.append(f"extern {c_ret_type} {func_name}({', '.join(c_arg_types) or 'void'});")
    lines.append("")
    lines.append("int main(int argc, char **argv) {")
    lines.append("    int ai = 1;")

    call_args = []
    for i, (kind, n) in enumerate(arg_specs):
        var = f"a{i}"
        if kind == "scalar":
            default = 10 * (i + 1)
            lines.append(f"    int {var} = (argc > ai) ? atoi(argv[ai++]) : {default};")
        else:
            lines.append(f"    v{n}si {var};")
            lines.append(f"    for (int k = 0; k < {n}; k++) {{")
            lines.append(f"        {var}[k] = (argc > ai) ? atoi(argv[ai++]) : (10 * {i} + k + 1);")
            lines.append("    }")
        call_args.append(var)

    call = f"{func_name}({', '.join(call_args)})"
    ret_kind, ret_n = ret_spec
    if ret_kind == "scalar":
        lines.append(f"    int r = {call};")
        lines.append('    printf("%d\\n", r);')
    else:
        lines.append(f"    v{ret_n}si r = {call};")
        lines.append('    printf("[");')
        lines.append(f"    for (int k = 0; k < {ret_n}; k++) {{")
        lines.append('        printf(k == 0 ? "%d" : ", %d", r[k]);')
        lines.append("    }")
        lines.append('    printf("]\\n");')

    lines.append("    return 0;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_harness(module, out_path, func_name: str = "my_func") -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(generate_harness_c(module, func_name))
