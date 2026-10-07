"""FileCheck tests for the C harness generator: each test parses a function
signature, generates the harness, and matches the C source against its
`// CHECK:` lines. Nothing here compiles C; the compiled-backend tests do.
"""
import pytest

from conftest import parse, filecheck
from back_end.harness_gen import generate_harness_c


def harness(ir: str) -> str:
    return generate_harness_c(parse(ir), "my_func")


def test_scalar_register_abi():
    """Without memrefs, `my_func` is called directly and scalars are passed
    as C `int`s. Each argument reads one CLI value, else a default."""
    filecheck(harness("""
    func.func @my_func(%a: i32, %b: i32) -> i32 {
      func.return %a : i32
    }"""), """
    // CHECK: extern int my_func(int, int);
    // CHECK: int a0 = (argc > ai) ? atoi(argv[ai++]) : 10;
    // CHECK-NEXT: int a1 = (argc > ai) ? atoi(argv[ai++]) : 20;
    // CHECK-NEXT: int r = my_func(a0, a1);
    """)


# A bufferized matmul (2x3 @ 3x4) plus a scalar argument. Only the
# signature matters to the generator.
MEMREF_FUNC = """
func.func @my_func(%a: memref<2x3xi32>, %b: memref<3x4xi32>, %s: i32) -> memref<2x4xi32> {
  %r = memref.alloc() : memref<2x4xi32>
  func.return %r : memref<2x4xi32>
}"""


def test_memref_descriptor_and_c_interface():
    """A memref is passed as a pointer to MLIR's descriptor struct, whose
    field order the callee relies on. `_mlir_ciface_my_func` returns the
    result through a leading out-parameter; the scalar stays by value."""
    filecheck(harness(MEMREF_FUNC), """
    // CHECK: typedef struct {
    // CHECK-NEXT: int32_t *allocated;
    // CHECK-NEXT: int32_t *aligned;
    // CHECK-NEXT: int64_t offset;
    // CHECK-NEXT: int64_t sizes[2];
    // CHECK-NEXT: int64_t strides[2];
    // CHECK-NEXT: } MemRef2D_i32;
    // CHECK: extern void _mlir_ciface_my_func(MemRef2D_i32 *result, MemRef2D_i32 *a0, MemRef2D_i32 *a1, int a2);
    """)


def test_memref_call_and_free():
    """Argument descriptors get their sizes and row-major strides
    ({cols, 1}). The result is printed through the strides the callee filled
    in, and its buffer, which the callee allocated, is freed."""
    filecheck(harness(MEMREF_FUNC), """
    // CHECK: MemRef2D_i32 a0 = { a0_data, a0_data, 0, {2, 3}, {3, 1} };
    // CHECK: MemRef2D_i32 a1 = { a1_data, a1_data, 0, {3, 4}, {4, 1} };
    // CHECK: int a2 = (argc > ai) ? atoi(argv[ai++]) : 30;
    // CHECK: _mlir_ciface_my_func(&result, &a0, &a1, a2);
    // CHECK: result.aligned[result.offset + i0 * result.strides[0] + i1 * result.strides[1]]
    // CHECK: free(result.allocated);
    // CHECK-NEXT: free(a0_data);
    // CHECK-NEXT: free(a1_data);
    """)


def test_rank_1_memref():
    """A rank-1 memref has one size and one stride, and the result is
    printed as a single row."""
    filecheck(harness("""
    func.func @my_func(%v: memref<4xi32>) -> memref<4xi32> {
      func.return %v : memref<4xi32>
    }"""), """
    // CHECK: int64_t sizes[1];
    // CHECK-NEXT: int64_t strides[1];
    // CHECK-NEXT: } MemRef1D_i32;
    // CHECK: extern void _mlir_ciface_my_func(MemRef1D_i32 *result, MemRef1D_i32 *a0);
    // CHECK: MemRef1D_i32 a0 = { a0_data, a0_data, 0, {4}, {1} };
    // CHECK: result.aligned[result.offset + i0 * result.strides[0]]
    """)


def test_rank_4_memref():
    """The descriptor follows the rank. Each stride is the product of the
    sizes after it: 3*4*5, 4*5, 5, 1."""
    filecheck(harness("""
    func.func @my_func(%t: memref<2x3x4x5xi32>) -> memref<2x3x4x5xi32> {
      func.return %t : memref<2x3x4x5xi32>
    }"""), """
    // CHECK: int64_t sizes[4];
    // CHECK-NEXT: int64_t strides[4];
    // CHECK-NEXT: } MemRef4D_i32;
    // CHECK: MemRef4D_i32 a0 = { a0_data, a0_data, 0, {2, 3, 4, 5}, {60, 20, 5, 1} };
    """)


def test_refuses_multi_output_function():
    """Both calling conventions assume one result, so a second one would be
    silently dropped."""
    with pytest.raises(RuntimeError, match="exactly 1 result"):
        harness("""
        func.func @my_func(%a: i32) -> (i32, i32) {
          func.return %a, %a : i32, i32
        }""")
