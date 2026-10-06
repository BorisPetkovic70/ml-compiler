"""FileCheck tests for constant CSE: each test parses a small function, runs
only `apply_constant_cse`, and matches the printed IR against its
`// CHECK:` lines.
"""
from conftest import parse, filecheck
from middle_end.cse import apply_constant_cse


def cse(ir: str):
    module = parse(ir)
    apply_constant_cse(module)
    module.verify()
    return module


def test_equal_constants_are_merged():
    """`%a` and `%b` are the same constant, so `%b` is erased and its user
    reads `%a` instead."""
    filecheck(cse("""
    func.func @f(%x: i32) -> i32 {
      %a = arith.constant 3 : i32
      %s = arith.addi %x, %a : i32
      %b = arith.constant 3 : i32
      %r = arith.muli %s, %b : i32
      func.return %r : i32
    }"""), """
    // CHECK: %a = arith.constant 3 : i32
    // CHECK-NEXT: %s = arith.addi %x, %a : i32
    // CHECK-NEXT: %r = arith.muli %s, %a : i32
    // CHECK-NEXT: func.return %r : i32
    """)


def test_constant_in_a_loop_body_is_hoisted_and_merged():
    """The loop body's own `0 : i32` is the same constant as `%zero`. A value
    defined at the top of the function is visible inside the loop, so the
    body uses `%zero` and its copy is erased."""
    filecheck(cse("""
    func.func @f(%m: memref<4xi32>) -> i32 {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 4 : index
      %zero = arith.constant 0 : i32
      scf.for %i = %c0 to %n step %c1 {
        %inner = arith.constant 0 : i32
        memref.store %inner, %m[%i] : memref<4xi32>
      }
      func.return %zero : i32
    }"""), """
    // CHECK: %zero = arith.constant 0 : i32
    // CHECK-NEXT: scf.for %i = %c0 to %n step %c1 {
    // CHECK-NEXT: memref.store %zero, %m[%i] : memref<4xi32>
    // CHECK-NEXT: }
    """)


def test_same_value_of_different_types_is_kept_apart():
    """`0 : index` and `0 : i32` are different constants, and both stay. The
    constants are moved to the top of the function."""
    filecheck(cse("""
    func.func @f(%m: memref<4xi32>) -> memref<4xi32> {
      %i = arith.constant 0 : index
      %v = arith.constant 0 : i32
      memref.store %v, %m[%i] : memref<4xi32>
      func.return %m : memref<4xi32>
    }"""), """
    // CHECK: %i = arith.constant 0 : index
    // CHECK-NEXT: %v = arith.constant 0 : i32
    // CHECK-NEXT: memref.store %v, %m[%i] : memref<4xi32>
    """)
