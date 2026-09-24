"""The generated C harness must match the compiled function's real ABI.

Nothing here compiles C (the suite stays toolchain-free), so these assert on the
*substance* of the generated source -- descriptor layout, argument order, stride
values -- not merely that some expected text appears. A harness with the right
shape but wrong strides would link fine and silently print garbage.
"""
import pytest

from xdsl.dialects import arith, func
from xdsl.dialects.builtin import MemRefType, i32
from xdsl.ir import Block, Region

from conftest import build_module, tensor_ty, lower, const_i32
from hc_dialect import HCMatmul, HCAdd
from back_end.harness_gen import generate_harness_c

# distinct M/K/N so a swapped dimension or stride can't hide
_M, _K, _N = 2, 3, 4


def _bufferized_matmul_harness() -> str:
    def body(args):
        a, b = args
        op = HCMatmul(operands=[a, b], result_types=[tensor_ty(_M, _N)])
        return [op], op.results[0]
    m = build_module([tensor_ty(_M, _K), tensor_ty(_K, _N)], body)
    lower(m, bufferize=True)
    return generate_harness_c(m, "my_func")


def test_memref_harness_calls_the_c_interface_wrapper_not_the_raw_function():
    """A memref signature must go through _mlir_ciface_* (the wrapper emitted by
    --llvm-request-c-wrappers); calling the raw symbol would pass an exploded
    descriptor as N separate scalars and corrupt the call."""
    src = _bufferized_matmul_harness()
    assert "_mlir_ciface_my_func" in src
    assert "extern void _mlir_ciface_my_func(" in src   # memref result -> void + out-param


def test_memref_result_is_the_leading_out_parameter():
    """MLIR's C interface turns a memref return into a *first* argument; getting the
    order wrong would write the result over an input descriptor."""
    src = _bufferized_matmul_harness()
    decl = next(l for l in src.splitlines() if l.startswith("extern void _mlir_ciface_"))
    params = decl[decl.index("(") + 1:decl.rindex(")")].split(", ")
    assert params[0].endswith("*result")
    assert [p.split()[-1] for p in params[1:]] == ["*a0", "*a1"]
    call = next(l for l in src.splitlines() if "_mlir_ciface_my_func(&result" in l)
    assert "&result, &a0, &a1" in call


def test_descriptor_struct_matches_the_mlir_memref_layout():
    """{allocated, aligned, offset, sizes[rank], strides[rank]} in that exact order --
    the struct is reinterpreted by the callee, so field order is load-bearing."""
    src = _bufferized_matmul_harness()
    body = src[src.index("typedef struct"):src.index("} MemRef2D_i32;")]
    fields = [l.strip() for l in body.splitlines()[1:] if l.strip()]
    assert fields == [
        "int32_t *allocated;",
        "int32_t *aligned;",
        "int64_t offset;",
        "int64_t sizes[2];",
        "int64_t strides[2];",
    ]


def test_argument_descriptors_carry_correct_sizes_and_row_major_strides():
    """Strides are the subtle part: row-major [d1, 1] for a 2-D buffer. A transposed
    or all-ones stride still compiles and still prints numbers -- just wrong ones."""
    src = _bufferized_matmul_harness()
    a0 = next(l for l in src.splitlines() if l.strip().startswith("MemRef2D_i32 a0 ="))
    a1 = next(l for l in src.splitlines() if l.strip().startswith("MemRef2D_i32 a1 ="))
    # A is MxK -> sizes {M, K}, strides {K, 1};  B is KxN -> sizes {K, N}, strides {N, 1}
    assert f"{{{_M}, {_K}}}" in a0 and f"{{{_K}, 1}}" in a0
    assert f"{{{_K}, {_N}}}" in a1 and f"{{{_N}, 1}}" in a1


def test_argument_buffers_are_allocated_element_count_and_freed():
    src = _bufferized_matmul_harness()
    assert f"malloc({_M * _K} * sizeof(int32_t))" in src
    assert f"malloc({_K * _N} * sizeof(int32_t))" in src
    # the callee mallocs the result, so the harness owns and frees it too
    assert "free(result.allocated);" in src
    assert "free(a0_data);" in src and "free(a1_data);" in src


def test_memref_harness_reads_cli_args_with_defaults_like_the_register_abi_one():
    """Same argc/argv-with-fallback pattern as the scalar/vector generator: main
    takes argv, and each element -- memref or scalar -- reads one CLI arg if given,
    falling back to the same default formula used when none is."""
    src = _bufferized_matmul_harness()
    assert "int main(int argc, char **argv) {" in src
    assert "int main(void)" not in src
    assert "int ai = 1;" in src
    # one memref element per CLI arg, in order, same fallback as the no-arg default
    assert (
        "a0_data[k] = (argc > ai) ? (int32_t)atoi(argv[ai++]) : (int32_t)(10 * 0 + k + 1);"
        in src
    )
    assert (
        "a1_data[k] = (argc > ai) ? (int32_t)atoi(argv[ai++]) : (int32_t)(10 * 1 + k + 1);"
        in src
    )


def test_memref_harness_scalar_argument_also_reads_cli_args_with_fallback():
    def build(args):
        return [], args[0]
    m = _module_with_signature([_memref_ty(2, 2), i32], build)
    src = generate_harness_c(m, "my_func")
    assert "int a1 = (argc > ai) ? atoi(argv[ai++]) : 20;" in src


def test_result_is_printed_through_its_own_descriptor_not_assumed_dims():
    """Printing must read sizes/strides back out of the returned descriptor rather
    than hardcoding them -- the callee is what filled them in."""
    src = _bufferized_matmul_harness()
    assert "result.sizes[0]" in src and "result.sizes[1]" in src
    assert "result.strides[0]" in src and "result.strides[1]" in src
    assert "result.offset" in src


# ---- the untouched scalar/vector path must keep its own (different) ABI ----------

def test_scalar_vector_harness_still_calls_the_raw_symbol_directly():
    """Scalar/vector models use the register ABI and call my_func directly; they must
    not be routed through the memref C-interface path."""
    def body(args):
        a, b = args
        op = HCAdd(operands=[a, b], result_types=[i32])
        return [op], op.results[0]
    m = build_module([i32, i32], body)
    src = generate_harness_c(m, "my_func")
    assert "_mlir_ciface" not in src
    assert "extern int my_func(int, int);" in src
    assert "MemRef" not in src


# ---- refusals: fail loudly rather than emit a harness that silently mismatches ----

def _memref_ty(*dims, width: int = 32):
    from xdsl.dialects.builtin import IntegerType
    return MemRefType(IntegerType(width), list(dims))


def _module_with_signature(arg_types, ret_op_builder):
    block = Block(arg_types=list(arg_types))
    ops, ret = ret_op_builder(list(block.args))
    for op in ops:
        block.add_op(op)
    block.add_op(func.ReturnOp(ret))
    fn = func.FuncOp("my_func", (list(arg_types), [ret.type]), region=Region(block))
    from xdsl.dialects.builtin import ModuleOp
    return ModuleOp(ops=[fn])


def test_refuses_memref_args_with_a_scalar_result():
    """The out-parameter convention assumes a memref result; a scalar result would
    make the emitted signature wrong in a way the linker cannot catch."""
    def build(args):
        c = const_i32(0)
        return [c], c.result
    m = _module_with_signature([_memref_ty(2, 2)], build)
    with pytest.raises(RuntimeError, match="must also return a memref"):
        generate_harness_c(m, "my_func")


def test_refuses_rank_3_memref():
    def build(args):
        return [], args[0]
    m = _module_with_signature([_memref_ty(2, 2, 2)], build)
    with pytest.raises(RuntimeError, match="rank-1 and rank-2"):
        generate_harness_c(m, "my_func")


def test_refuses_unsupported_element_type():
    def build(args):
        return [], args[0]
    m = _module_with_signature([_memref_ty(2, 2, width=7)], build)
    with pytest.raises(RuntimeError, match="unsupported memref element type"):
        generate_harness_c(m, "my_func")


def test_vector_harness_uses_the_register_abi_typedef():
    """The vector path predates this file's tests; pin it now that the known-working
    compiled vector ABI (GCC vector_size, passed in registers) depends on it."""
    from conftest import const_vec
    from hc_dialect import HCAddVec

    def body(args):
        v = const_vec([1, 2, 3, 4])
        op = HCAddVec(operands=[args[0], v.result], result_types=[args[0].type])
        return [v, op], op.results[0]
    from conftest import vec_ty
    m = build_module([vec_ty(4)], body)
    src = generate_harness_c(m, "my_func")
    assert "typedef int v4si __attribute__((vector_size(16)));" in src
    assert "extern v4si my_func(v4si);" in src


def test_scalar_argument_alongside_memref_is_passed_by_value():
    """Only memrefs become pointers in the C interface; a scalar stays by value."""
    def build(args):
        return [], args[0]
    m = _module_with_signature([_memref_ty(2, 2), i32], build)
    src = generate_harness_c(m, "my_func")
    decl = next(l for l in src.splitlines() if l.startswith("extern void _mlir_ciface_"))
    params = [p.strip() for p in decl[decl.index("(") + 1:decl.rindex(")")].split(",")]
    assert params[1].endswith("*a0")      # memref -> pointer
    assert params[2] == "int a1"          # scalar -> by value
    assert "&result, &a0, a1" in src      # and passed without '&'


def test_rank_1_memref_result_prints_as_a_flat_row():
    def build(args):
        return [], args[0]
    m = _module_with_signature([_memref_ty(4)], build)
    src = generate_harness_c(m, "my_func")
    assert "int64_t sizes[1];" in src
    assert "result.sizes[0]" in src and "result.strides[1]" not in src


def test_refuses_mixing_vector_and_memref_arguments():
    from conftest import vec_ty

    def build(args):
        return [], args[0]
    m = _module_with_signature([_memref_ty(2, 2), vec_ty(4)], build)
    with pytest.raises(RuntimeError, match="mixing vector and memref"):
        generate_harness_c(m, "my_func")


def test_write_harness_writes_the_generated_source(tmp_path):
    def body(args):
        a, b = args
        op = HCAdd(operands=[a, b], result_types=[i32])
        return [op], op.results[0]
    m = build_module([i32, i32], body)
    out = tmp_path / "h.c"
    from back_end.harness_gen import write_harness
    write_harness(m, out, func_name="my_func")
    assert out.read_text() == generate_harness_c(m, "my_func")
