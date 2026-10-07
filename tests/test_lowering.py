"""FileCheck tests for the lowering pass: each test parses a small `hc`
function, runs only lowering, and matches the printed IR against its
`// CHECK:` lines.
"""
import pytest

from conftest import parse, lower, filecheck


def check_lowering(ir: str, checks: str) -> None:
    """Lowers `ir`, matches `checks`, and checks that no `hc.` op survives."""
    module = lower(parse(ir))
    filecheck(module, checks)
    # A CHECK-NOT on its own scans the whole output.
    filecheck(module, "// CHECK-NOT: hc.")


@pytest.mark.parametrize("hc_op,arith_op", [
    ("hc.add", "arith.addi"), ("hc.sub", "arith.subi"), ("hc.mul", "arith.muli"),
])
def test_scalar_binop(hc_op, arith_op):
    check_lowering(f"""
    func.func @f(%a: i32, %b: i32) -> i32 {{
      %r = "{hc_op}"(%a, %b) : (i32, i32) -> i32
      func.return %r : i32
    }}""", f"""
    // CHECK: %[[R:.*]] = {arith_op} %a, %b : i32
    // CHECK: func.return %[[R]] : i32
    """)


def test_relu():
    check_lowering("""
    func.func @f(%x: i32) -> i32 {
      %r = "hc.relu"(%x) : (i32) -> i32
      func.return %r : i32
    }""", """
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: %[[R:.*]] = arith.maxsi %x, %[[ZERO]] : i32
    // CHECK: func.return %[[R]] : i32
    """)


@pytest.mark.parametrize("hc_op,predicate", [("hc.max", "sgt"), ("hc.min", "slt")])
def test_max_min(hc_op, predicate):
    check_lowering(f"""
    func.func @f(%a: i32, %b: i32) -> i32 {{
      %r = "{hc_op}"(%a, %b) : (i32, i32) -> i32
      func.return %r : i32
    }}""", f"""
    // CHECK: %[[COND:.*]] = arith.cmpi {predicate}, %a, %b : i32
    // CHECK: %[[R:.*]] = scf.if %[[COND]] -> (i32)
    // CHECK-NEXT: scf.yield %a : i32
    // CHECK-NEXT: else
    // CHECK-NEXT: scf.yield %b : i32
    // CHECK: func.return %[[R]] : i32
    """)


def test_pow():
    check_lowering("""
    func.func @f(%base: i32, %exp: i32) -> i32 {
      %r = "hc.pow"(%base, %exp) : (i32, i32) -> i32
      func.return %r : i32
    }""", """
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[UB:.*]] = arith.index_cast %exp : i32 to index
    // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: %[[R:.*]] = scf.for %{{.*}} = %[[C0]] to %[[UB]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[ONE]]) -> (i32)
    // CHECK-NEXT: %[[NEXT:.*]] = arith.muli %[[ACC]], %base : i32
    // CHECK-NEXT: scf.yield %[[NEXT]] : i32
    // CHECK: func.return %[[R]] : i32
    """)


@pytest.mark.parametrize("hc_op,arith_op", [
    ("hc.add", "arith.addi"), ("hc.sub", "arith.subi"), ("hc.mul", "arith.muli"),
])
def test_tensor_binop(hc_op, arith_op):
    """One loop per dim: the i/j nest threads the result tensor through
    iter_args and writes C[i, j] = A[i, j] op B[i, j]. The result starts as
    `tensor.empty`, because the nest writes every element."""
    check_lowering(f"""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {{
      %r = "{hc_op}"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }}""", f"""
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[M:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[N:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<2x3xi32>
    // CHECK: %[[R:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[M]] step %[[C1]] iter_args(%[[TI:.*]] = %[[INIT]]) -> (tensor<2x3xi32>)
    // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[TJ:.*]] = %[[TI]]) -> (tensor<2x3xi32>)
    // CHECK: %[[X:.*]] = tensor.extract %a[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[Y:.*]] = tensor.extract %b[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[Z:.*]] = {arith_op} %[[X]], %[[Y]] : i32
    // CHECK: %[[T:.*]] = tensor.insert %[[Z]] into %[[TJ]][%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: scf.yield %[[T]] : tensor<2x3xi32>
    // CHECK: func.return %[[R]] : tensor<2x3xi32>
    """)


def test_tensor_binop_rank_1():
    """A rank-1 tensor needs a single loop and a single index."""
    check_lowering("""
    func.func @f(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
      %r = "hc.add"(%a, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
      func.return %r : tensor<4xi32>
    }""", """
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[N:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<4xi32>
    // CHECK: %[[R:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[T:.*]] = %[[INIT]]) -> (tensor<4xi32>)
    // CHECK-NOT: scf.for
    // CHECK: %[[X:.*]] = tensor.extract %a[%[[I]]] : tensor<4xi32>
    // CHECK: %[[Y:.*]] = tensor.extract %b[%[[I]]] : tensor<4xi32>
    // CHECK: %[[Z:.*]] = arith.addi %[[X]], %[[Y]] : i32
    // CHECK: %[[NEXT:.*]] = tensor.insert %[[Z]] into %[[T]][%[[I]]] : tensor<4xi32>
    // CHECK: scf.yield %[[NEXT]] : tensor<4xi32>
    // CHECK: func.return %[[R]] : tensor<4xi32>
    """)


def test_broadcast_aligns_operands_at_the_last_dim():
    """`%b` has one dim, which lines up with the result's last one, so it is
    indexed by the inner induction variable only."""
    check_lowering("""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<3xi32>) -> tensor<2x3xi32> {
      %r = "hc.add"(%a, %b) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }""", """
    // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
    // CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
    // CHECK: %[[X:.*]] = tensor.extract %a[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[Y:.*]] = tensor.extract %b[%[[J]]] : tensor<3xi32>
    // CHECK: %[[Z:.*]] = arith.addi %[[X]], %[[Y]] : i32
    // CHECK: tensor.insert %[[Z]] into %{{.*}}[%[[I]], %[[J]]] : tensor<2x3xi32>
    """)


def test_broadcast_reads_a_size_1_dim_at_index_0():
    """`%b` has one column and the result three, so every `j` reads column
    0 of `%b`."""
    check_lowering("""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<2x1xi32>) -> tensor<2x3xi32> {
      %r = "hc.add"(%a, %b) : (tensor<2x3xi32>, tensor<2x1xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }""", """
    // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
    // CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
    // CHECK: %[[X:.*]] = tensor.extract %a[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
    // CHECK: %[[Y:.*]] = tensor.extract %b[%[[I]], %[[ZERO]]] : tensor<2x1xi32>
    // CHECK: %[[Z:.*]] = arith.addi %[[X]], %[[Y]] : i32
    // CHECK: tensor.insert %[[Z]] into %{{.*}}[%[[I]], %[[J]]] : tensor<2x3xi32>
    """)


def test_broadcast_uses_a_scalar_operand_directly():
    """A scalar has nothing to index: `%s` goes straight into the multiply."""
    check_lowering("""
    func.func @f(%s: i32, %a: tensor<2x3xi32>) -> tensor<2x3xi32> {
      %r = "hc.mul"(%s, %a) : (i32, tensor<2x3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }""", """
    // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
    // CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
    // CHECK-NEXT: %[[X:.*]] = tensor.extract %a[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK-NEXT: %[[Z:.*]] = arith.muli %s, %[[X]] : i32
    // CHECK-NEXT: tensor.insert %[[Z]] into %{{.*}}[%[[I]], %[[J]]] : tensor<2x3xi32>
    """)


def test_max_on_tensors_puts_the_scf_if_in_the_loop_body():
    """The same `arith.cmpi` + `scf.if` as scalar `hc.max`, applied to the
    two extracted elements."""
    check_lowering("""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<3xi32>) -> tensor<2x3xi32> {
      %r = "hc.max"(%a, %b) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }""", """
    // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
    // CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
    // CHECK-NEXT: %[[X:.*]] = tensor.extract %a[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK-NEXT: %[[Y:.*]] = tensor.extract %b[%[[J]]] : tensor<3xi32>
    // CHECK-NEXT: %[[COND:.*]] = arith.cmpi sgt, %[[X]], %[[Y]] : i32
    // CHECK-NEXT: %[[Z:.*]] = scf.if %[[COND]] -> (i32)
    // CHECK-NEXT: scf.yield %[[X]] : i32
    // CHECK-NEXT: else
    // CHECK-NEXT: scf.yield %[[Y]] : i32
    // CHECK: tensor.insert %[[Z]] into %{{.*}}[%[[I]], %[[J]]] : tensor<2x3xi32>
    """)


def test_equal_loop_bounds_share_one_constant():
    """A 2x2 result needs the bound 2 twice; both loops use the same
    constant."""
    check_lowering("""
    func.func @f(%x: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %r = "hc.relu"(%x) : (tensor<2x2xi32>) -> tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    }""", """
    // CHECK: %[[N:.*]] = arith.constant 2 : index
    // CHECK-NOT: arith.constant 2 : index
    // CHECK: scf.for %{{.*}} = %{{.*}} to %[[N]] step
    // CHECK: scf.for %{{.*}} = %{{.*}} to %[[N]] step
    """)


def test_relu_tensor():
    check_lowering("""
    func.func @f(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
      %r = "hc.relu"(%x) : (tensor<2x3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }""", """
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[M:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[N:.*]] = arith.constant 3 : index
    // CHECK: %[[R:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[M]] step %[[C1]] iter_args({{.*}}) -> (tensor<2x3xi32>)
    // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[TJ:.*]] = {{.*}}) -> (tensor<2x3xi32>)
    // CHECK: %[[X:.*]] = tensor.extract %x[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: %[[Y:.*]] = arith.maxsi %[[X]], %[[ZERO]] : i32
    // CHECK: tensor.insert %[[Y]] into %[[TJ]][%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: func.return %[[R]] : tensor<2x3xi32>
    """)


def test_matmul():
    """An i/j/k loop nest: the k loop accumulates A[i, k] * B[k, j] in an i32
    iter_arg, and the j loop inserts the sum at C[i, j]. M, K and N differ so
    a swapped bound or index shows up."""
    check_lowering("""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<3x4xi32>) -> tensor<2x4xi32> {
      %r = "hc.matmul"(%a, %b) : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
      func.return %r : tensor<2x4xi32>
    }""", """
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[M:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[N:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[K:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<2x4xi32>
    // CHECK: %[[R:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[M]] step %[[C1]] iter_args(%[[TI:.*]] = %[[INIT]]) -> (tensor<2x4xi32>)
    // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[TJ:.*]] = %[[TI]]) -> (tensor<2x4xi32>)
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: %[[SUM:.*]] = scf.for %[[KV:.*]] = %[[C0]] to %[[K]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (i32)
    // CHECK: %[[X:.*]] = tensor.extract %a[%[[I]], %[[KV]]] : tensor<2x3xi32>
    // CHECK: %[[Y:.*]] = tensor.extract %b[%[[KV]], %[[J]]] : tensor<3x4xi32>
    // CHECK: %[[P:.*]] = arith.muli %[[X]], %[[Y]] : i32
    // CHECK: %[[NEXT:.*]] = arith.addi %[[ACC]], %[[P]] : i32
    // CHECK: scf.yield %[[NEXT]] : i32
    // CHECK: tensor.insert %[[SUM]] into %[[TJ]][%[[I]], %[[J]]] : tensor<2x4xi32>
    // CHECK: func.return %[[R]] : tensor<2x4xi32>
    """)
