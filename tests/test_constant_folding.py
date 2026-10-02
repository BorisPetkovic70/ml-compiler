"""FileCheck tests for constant folding: each test parses a small `arith`
function, runs only the `FoldArithInts` pattern, and matches the printed IR
against its `// CHECK:` lines.
"""
from xdsl.pattern_rewriter import PatternRewriteWalker

from conftest import parse, filecheck
from middle_end.constant_folding import FoldArithInts


def fold(ir: str):
    """Runs only `FoldArithInts`. The constants it folded away stay in the
    IR, now unused; removing them is DCE's job. (The pipeline's greedy
    applier also erases dead ops, so they don't appear there.)"""
    module = parse(ir)
    PatternRewriteWalker(FoldArithInts()).rewrite_module(module)
    module.verify()
    return module


def test_scalar_chain():
    """(3 + 4) * 2: the add folds to 7, which makes the mul foldable too."""
    filecheck(fold("""
    func.func @f() -> i32 {
      %a = arith.constant 3 : i32
      %b = arith.constant 4 : i32
      %s = arith.addi %a, %b : i32
      %c = arith.constant 2 : i32
      %r = arith.muli %s, %c : i32
      func.return %r : i32
    }"""), """
    // CHECK: %s = arith.constant 7 : i32
    // CHECK: %r = arith.constant 14 : i32
    // CHECK-NEXT: func.return %r : i32
    """)


def test_vector_chain():
    """relu((a + b) * -1) folds lane by lane. Every product is negative, so
    the relu (`arith.maxsi` with 0) gives all zeros."""
    filecheck(fold("""
    func.func @f() -> vector<4xi32> {
      %a = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
      %b = arith.constant dense<[10, 20, 30, 40]> : vector<4xi32>
      %s = arith.addi %a, %b : vector<4xi32>
      %neg = arith.constant dense<-1> : vector<4xi32>
      %p = arith.muli %s, %neg : vector<4xi32>
      %zero = arith.constant dense<0> : vector<4xi32>
      %r = arith.maxsi %p, %zero : vector<4xi32>
      func.return %r : vector<4xi32>
    }"""), """
    // CHECK: %s = arith.constant dense<[11, 22, 33, 44]> : vector<4xi32>
    // CHECK: %p = arith.constant dense<[-11, -22, -33, -44]> : vector<4xi32>
    // CHECK: %r = arith.constant dense<0> : vector<4xi32>
    // CHECK-NEXT: func.return %r : vector<4xi32>
    """)


def test_broadcast_of_a_constant():
    """A broadcast of a constant scalar becomes a splat vector constant, so
    the mul that uses it folds as well."""
    filecheck(fold("""
    func.func @f() -> vector<4xi32> {
      %c = arith.constant 3 : i32
      %v = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
      %b = vector.broadcast %c : i32 to vector<4xi32>
      %r = arith.muli %v, %b : vector<4xi32>
      func.return %r : vector<4xi32>
    }"""), """
    // CHECK: %b = arith.constant dense<3> : vector<4xi32>
    // CHECK-NEXT: %r = arith.constant dense<[3, 6, 9, 12]> : vector<4xi32>
    // CHECK-NEXT: func.return %r : vector<4xi32>
    """)


def test_runtime_operand_is_not_folded():
    """`%x` is only known at run time, so the add stays."""
    filecheck(fold("""
    func.func @f(%x: i32) -> i32 {
      %c = arith.constant 4 : i32
      %r = arith.addi %x, %c : i32
      func.return %r : i32
    }"""), """
    // CHECK: %r = arith.addi %x, %c : i32
    // CHECK-NEXT: func.return %r : i32
    """)
