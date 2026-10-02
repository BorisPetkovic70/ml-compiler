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
    ("hc.add_vec", "arith.addi"), ("hc.sub_vec", "arith.subi"),
    ("hc.mul_vec_vec", "arith.muli"),
])
def test_vector_binop(hc_op, arith_op):
    check_lowering(f"""
    func.func @f(%a: vector<4xi32>, %b: vector<4xi32>) -> vector<4xi32> {{
      %r = "{hc_op}"(%a, %b) : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
      func.return %r : vector<4xi32>
    }}""", f"""
    // CHECK: %[[R:.*]] = {arith_op} %a, %b : vector<4xi32>
    // CHECK: func.return %[[R]] : vector<4xi32>
    """)


def test_mul_vec_broadcasts_the_scalar():
    check_lowering("""
    func.func @f(%s: i32, %v: vector<4xi32>) -> vector<4xi32> {
      %r = "hc.mul_vec"(%s, %v) : (i32, vector<4xi32>) -> vector<4xi32>
      func.return %r : vector<4xi32>
    }""", """
    // CHECK: %[[S:.*]] = vector.broadcast %s : i32 to vector<4xi32>
    // CHECK: %[[R:.*]] = arith.muli %v, %[[S]] : vector<4xi32>
    // CHECK: func.return %[[R]] : vector<4xi32>
    """)


def test_relu_vec():
    check_lowering("""
    func.func @f(%x: vector<4xi32>) -> vector<4xi32> {
      %r = "hc.relu_vec"(%x) : (vector<4xi32>) -> vector<4xi32>
      func.return %r : vector<4xi32>
    }""", """
    // CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<4xi32>
    // CHECK: %[[R:.*]] = arith.maxsi %x, %[[ZERO]] : vector<4xi32>
    // CHECK: func.return %[[R]] : vector<4xi32>
    """)


@pytest.mark.parametrize("hc_op,arith_op", [
    ("hc.add_tensor", "arith.addi"), ("hc.sub_tensor", "arith.subi"),
    ("hc.mul_tensor", "arith.muli"),
])
def test_tensor_binop(hc_op, arith_op):
    """An i/j loop nest threads the result tensor through iter_args and
    writes C[i, j] = A[i, j] op B[i, j]."""
    check_lowering(f"""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {{
      %r = "{hc_op}"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }}""", f"""
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[M:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[N:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0> : tensor<2x3xi32>
    // CHECK: %[[R:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[M]] step %[[C1]] iter_args(%[[TI:.*]] = %[[INIT]]) -> (tensor<2x3xi32>)
    // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[TJ:.*]] = %[[TI]]) -> (tensor<2x3xi32>)
    // CHECK: %[[X:.*]] = tensor.extract %a[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[Y:.*]] = tensor.extract %b[%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: %[[Z:.*]] = {arith_op} %[[X]], %[[Y]] : i32
    // CHECK: %[[T:.*]] = tensor.insert %[[Z]] into %[[TJ]][%[[I]], %[[J]]] : tensor<2x3xi32>
    // CHECK: scf.yield %[[T]] : tensor<2x3xi32>
    // CHECK: func.return %[[R]] : tensor<2x3xi32>
    """)


def test_relu_tensor():
    check_lowering("""
    func.func @f(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
      %r = "hc.relu_tensor"(%x) : (tensor<2x3xi32>) -> tensor<2x3xi32>
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
    // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0> : tensor<2x4xi32>
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
