"""Generate a C harness matching an hc module's entry function signature.

Since the compiled pipeline calls the emitted LLVM function directly through a
C `extern` declaration, the harness has to match that function's exact
argument/result types -- which vary per model (scalar-only, vector-only,
multiple inputs, mixed scalar+vector). One static harness.c can't cover all of
them, so this generates a matching one per model instead.

There are two calling conventions to match, so there are two generators:

  * scalar/vector signatures call the emitted function `<fn>` directly, passing
    scalars as `int` and vectors as GCC `vector_size` values (register ABI);
  * any signature containing a `memref` (i.e. anything that went through
    bufferization) instead calls `_mlir_ciface_<fn>`, the C wrapper emitted by
    mlir-opt's --llvm-request-c-wrappers. There, every memref is passed as a
    *pointer to a descriptor struct* {allocated, aligned, offset, sizes[],
    strides[]}, and a memref return value becomes a leading out-parameter:

        void _mlir_ciface_my_func(MemRef2D_i32 *result, MemRef2D_i32 *a, ...)

    The callee mallocs the returned buffer (memref.alloc lowers to malloc), so
    the harness frees result.allocated when it is done.

Because the memref signature only appears after bufferization, hc_main.py
generates the harness *after* running the middle end, not before it.
"""
from xdsl.dialects import func
from xdsl.dialects.builtin import MemRefType, VectorType


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
    in_types = list(fn.function_type.inputs.data)
    out_types = list(fn.function_type.outputs.data)
    if len(out_types) != 1:
        raise RuntimeError("harness generator expects exactly 1 result")

    if any(isinstance(t, MemRefType) for t in in_types + out_types):
        return _generate_memref_harness(func_name, in_types, out_types[0])
    return _generate_scalar_vector_harness(func_name, in_types, out_types[0])


def _generate_scalar_vector_harness(func_name, in_types, out_type) -> str:
    arg_specs = [_type_spec(t) for t in in_types]
    out_types = [out_type]
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


# -----------------------------------------------------------------------------
#  memref C-ABI harness (_mlir_ciface_<fn>, descriptor structs)
# -----------------------------------------------------------------------------
_C_INT_BY_WIDTH = {8: "int8_t", 16: "int16_t", 32: "int32_t", 64: "int64_t"}


def _memref_dims(ty: MemRefType) -> list[int]:
    return [int(getattr(d, "data", d)) for d in ty.shape]


def _memref_elem_width(ty: MemRefType) -> int:
    width = getattr(ty.element_type, "width", None)
    width = int(getattr(width, "data", width)) if width is not None else None
    if width not in _C_INT_BY_WIDTH:
        raise RuntimeError(
            f"harness generator: unsupported memref element type {ty.element_type} "
            f"(integer widths {sorted(_C_INT_BY_WIDTH)} only)"
        )
    return width


def _descriptor_name(rank: int, width: int) -> str:
    return f"MemRef{rank}D_i{width}"


def _row_major_strides(dims: list[int]) -> list[int]:
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
    return strides


def _print_result_lines(out_dims: list[int]) -> list[str]:
    """Nested for-loops printing result.aligned in bracketed rows: every
    dimension but the last wraps a for-loop that prints '[' on entry and ']'
    plus a newline on exit; the last dimension is the flat, comma-separated
    printf loop over individual elements. A rank-1 result has no wrapping
    loop at all -- just the single bracketed line -- and each additional
    dimension adds one more wrapping loop around that same innermost line."""
    rank = len(out_dims)
    ivs = [f"i{d}" for d in range(rank)]
    idx = "result.offset + " + " + ".join(
        f"{iv} * result.strides[{d}]" for d, iv in enumerate(ivs)
    )
    last = ivs[-1]

    lines = []
    indent = "    "
    for d in range(rank - 1):
        lines.append(f"{indent}for (int64_t {ivs[d]} = 0; {ivs[d]} < result.sizes[{d}]; {ivs[d]}++) {{")
        indent += "    "
    lines.append(f'{indent}printf("[");')
    lines.append(f"{indent}for (int64_t {last} = 0; {last} < result.sizes[{rank - 1}]; {last}++)")
    lines.append(f'{indent}    printf({last} ? ", %d" : "%d", (int)result.aligned[{idx}]);')
    lines.append(f'{indent}printf("]\\n");')
    for _ in range(rank - 1):
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    return lines


def _generate_memref_harness(func_name, in_types, out_type) -> str:
    if not isinstance(out_type, MemRefType):
        raise RuntimeError(
            "harness generator: a memref-taking function must also return a memref "
            f"(got {out_type}); the C wrapper's out-parameter convention assumes it"
        )
    for t in in_types:
        if not isinstance(t, MemRefType) and isinstance(t, VectorType):
            raise RuntimeError(
                "harness generator: mixing vector and memref arguments is not supported"
            )

    memref_types = [t for t in in_types if isinstance(t, MemRefType)] + [out_type]
    for t in memref_types:
        if len(_memref_dims(t)) not in (1, 2, 3):
            raise RuntimeError(
                f"harness generator: only rank-1, rank-2, and rank-3 (batched) "
                f"memrefs are supported, got {t}"
            )

    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#include <stdint.h>", ""]

    # One descriptor struct per distinct (rank, element width) actually used.
    shapes = sorted({(len(_memref_dims(t)), _memref_elem_width(t)) for t in memref_types})
    for rank, width in shapes:
        elem = _C_INT_BY_WIDTH[width]
        lines += [
            f"typedef struct {{",
            f"    {elem} *allocated;",
            f"    {elem} *aligned;",
            f"    int64_t offset;",
            f"    int64_t sizes[{rank}];",
            f"    int64_t strides[{rank}];",
            f"}} {_descriptor_name(rank, width)};",
            "",
        ]

    # _mlir_ciface_<fn>: memref result becomes a leading out-parameter, every memref
    # argument is passed by pointer, scalars stay by value.
    res_desc = _descriptor_name(len(_memref_dims(out_type)), _memref_elem_width(out_type))
    params = [f"{res_desc} *result"]
    for i, t in enumerate(in_types):
        if isinstance(t, MemRefType):
            params.append(f"{_descriptor_name(len(_memref_dims(t)), _memref_elem_width(t))} *a{i}")
        else:
            params.append(f"int a{i}")
    lines.append(f"extern void _mlir_ciface_{func_name}({', '.join(params)});")
    lines += ["", "int main(int argc, char **argv) {", "    int ai = 1;"]

    call_args = ["&result"]
    owned = []
    for i, t in enumerate(in_types):
        if not isinstance(t, MemRefType):
            default = 10 * (i + 1)
            lines.append(f"    int a{i} = (argc > ai) ? atoi(argv[ai++]) : {default};")
            call_args.append(f"a{i}")
            continue
        dims, width = _memref_dims(t), _memref_elem_width(t)
        elem, desc = _C_INT_BY_WIDTH[width], _descriptor_name(len(dims), width)
        count = 1
        for d in dims:
            count *= d
        lines += [
            f"    /* argument {i}: memref<{'x'.join(str(d) for d in dims)}xi{width}> */",
            f"    {elem} *a{i}_data = ({elem} *)malloc({count} * sizeof({elem}));",
            f"    for (int64_t k = 0; k < {count}; k++)",
            f"        a{i}_data[k] = (argc > ai) ? ({elem})atoi(argv[ai++]) : ({elem})(10 * {i} + k + 1);",
            f"    {desc} a{i} = {{ a{i}_data, a{i}_data, 0,"
            f" {{{', '.join(str(d) for d in dims)}}},"
            f" {{{', '.join(str(s) for s in _row_major_strides(dims))}}} }};",
        ]
        call_args.append(f"&a{i}")
        owned.append(f"a{i}_data")

    out_dims = _memref_dims(out_type)
    lines += [
        "",
        f"    /* the callee allocates the result buffer, so this descriptor is filled in */",
        f"    {res_desc} result;",
        f"    _mlir_ciface_{func_name}({', '.join(call_args)});",
        "",
    ]

    lines += _print_result_lines(out_dims)

    lines.append("")
    lines.append("    free(result.allocated);")
    for name in owned:
        lines.append(f"    free({name});")
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


def write_harness(module, out_path, func_name: str = "my_func") -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(generate_harness_c(module, func_name))
