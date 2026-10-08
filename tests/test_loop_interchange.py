"""FileCheck tests for loop interchange: each test parses a small bufferized
function, runs only `apply_loop_interchange`, and matches the printed IR
against its `// CHECK:` lines.
"""
from conftest import parse, filecheck
from middle_end.loop_interchange import apply_loop_interchange


def interchange(ir: str):
    module = parse(ir)
    apply_loop_interchange(module)
    module.verify()
    return module


def test_matmul_nest_becomes_ikj():
    """The body updates `%C[%i, %j]` in place, so the `%j` and `%k` loops
    swap. Each loop keeps its own bound (`%N` for `%j`, `%K` for `%k`) and
    the body is unchanged."""
    filecheck(interchange("""
    func.func @f(%A: memref<2x3xi32>, %B: memref<3x4xi32>, %C: memref<2x4xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 2 : index
      %N = arith.constant 4 : index
      %K = arith.constant 3 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %j = %c0 to %N step %c1 {
          scf.for %k = %c0 to %K step %c1 {
            %c = memref.load %C[%i, %j] : memref<2x4xi32>
            %a = memref.load %A[%i, %k] : memref<2x3xi32>
            %b = memref.load %B[%k, %j] : memref<3x4xi32>
            %p = arith.muli %a, %b : i32
            %s = arith.addi %c, %p : i32
            memref.store %s, %C[%i, %j] : memref<2x4xi32>
          }
        }
      }
      func.return
    }"""), """
    // CHECK: scf.for %i = %c0 to %M step %c1 {
    // CHECK-NEXT: scf.for %k = %c0 to %K step %c1 {
    // CHECK-NEXT: scf.for %j = %c0 to %N step %c1 {
    // CHECK-NEXT: %c = memref.load %C[%i, %j] : memref<2x4xi32>
    // CHECK-NEXT: %a = memref.load %A[%i, %k] : memref<2x3xi32>
    // CHECK-NEXT: %b = memref.load %B[%k, %j] : memref<3x4xi32>
    // CHECK-NEXT: %p = arith.muli %a, %b : i32
    // CHECK-NEXT: %s = arith.addi %c, %p : i32
    // CHECK-NEXT: memref.store %s, %C[%i, %j] : memref<2x4xi32>
    """)


def test_elementwise_nest_is_left_unchanged():
    """The body stores to `%R` and never loads it, so it is not an in-place
    update and the loops stay in `%i`, `%j` order."""
    filecheck(interchange("""
    func.func @f(%A: memref<2x3xi32>, %B: memref<2x3xi32>, %R: memref<2x3xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 2 : index
      %N = arith.constant 3 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %j = %c0 to %N step %c1 {
          %a = memref.load %A[%i, %j] : memref<2x3xi32>
          %b = memref.load %B[%i, %j] : memref<2x3xi32>
          %s = arith.addi %a, %b : i32
          memref.store %s, %R[%i, %j] : memref<2x3xi32>
        }
      }
      func.return
    }"""), """
    // CHECK: scf.for %i = %c0 to %M step %c1 {
    // CHECK-NEXT: scf.for %j = %c0 to %N step %c1 {
    // CHECK-NEXT: %a = memref.load %A[%i, %j] : memref<2x3xi32>
    """)
