"""FileCheck tests for dead code elimination: each test parses a small
function, runs only `apply_dce`, and matches the printed IR against its
`// CHECK:` lines. The CHECK-NEXT chains show that nothing else survives.
"""
from conftest import parse, filecheck
from middle_end.dead_code_elimination import apply_dce


def dce(ir: str):
    module = parse(ir)
    apply_dce(module)
    module.verify()
    return module


def test_dead_chain_is_removed_and_used_ops_stay():
    """`%dead` has no uses, and once it is gone neither do `%a` and `%b`.
    `%one` and `%r` feed the return, so they stay."""
    filecheck(dce("""
    func.func @f(%x: i32) -> i32 {
      %a = arith.constant 3 : i32
      %b = arith.constant 4 : i32
      %dead = arith.addi %a, %b : i32
      %one = arith.constant 1 : i32
      %r = arith.addi %x, %one : i32
      func.return %r : i32
    }"""), """
    // CHECK: func.func @f(%x: i32) -> i32 {
    // CHECK-NEXT: %one = arith.constant 1 : i32
    // CHECK-NEXT: %r = arith.addi %x, %one : i32
    // CHECK-NEXT: func.return %r : i32
    """)


def test_unused_broadcast_is_removed():
    """Constant folding can't remove this broadcast, because `%x` is only
    known at run time. Its result is unused, so DCE does."""
    filecheck(dce("""
    func.func @f(%x: i32) -> i32 {
      %v = vector.broadcast %x : i32 to vector<4xi32>
      func.return %x : i32
    }"""), """
    // CHECK: func.func @f(%x: i32) -> i32 {
    // CHECK-NEXT: func.return %x : i32
    """)
