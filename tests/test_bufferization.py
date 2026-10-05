"""FileCheck tests for bufferization: each test parses already-lowered tensor
IR, runs only `apply_bufferization`, and matches the printed IR against its
`// CHECK:` lines.
"""
import pytest

from conftest import parse, filecheck
from middle_end.bufferization import apply_bufferization


def bufferize(ir: str):
    module = parse(ir)
    apply_bufferization(module)
    module.verify()
    return module


def check_bufferization(ir: str, checks: str) -> None:
    """Bufferizes `ir`, matches `checks`, and checks that no `tensor.` op
    survives."""
    module = bufferize(ir)
    filecheck(module, checks)
    # A CHECK-NOT on its own scans the whole output.
    filecheck(module, "// CHECK-NOT: tensor.")


# hc.matmul (2x3 @ 3x4) after lowering, with readable value names.
MATMUL = """
func.func @matmul(%a: tensor<2x3xi32>, %b: tensor<3x4xi32>) -> tensor<2x4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = arith.constant 2 : index
  %n = arith.constant 4 : index
  %kd = arith.constant 3 : index
  %init = arith.constant dense<0> : tensor<2x4xi32>
  %r = scf.for %i = %c0 to %m step %c1 iter_args(%ti = %init) -> (tensor<2x4xi32>) {
    %rj = scf.for %j = %c0 to %n step %c1 iter_args(%tj = %ti) -> (tensor<2x4xi32>) {
      %zero = arith.constant 0 : i32
      %sum = scf.for %k = %c0 to %kd step %c1 iter_args(%acc = %zero) -> (i32) {
        %x = tensor.extract %a[%i, %k] : tensor<2x3xi32>
        %y = tensor.extract %b[%k, %j] : tensor<3x4xi32>
        %p = arith.muli %x, %y : i32
        %next = arith.addi %acc, %p : i32
        scf.yield %next : i32
      }
      %t = tensor.insert %sum into %tj[%i, %j] : tensor<2x4xi32>
      scf.yield %t : tensor<2x4xi32>
    }
    scf.yield %rj : tensor<2x4xi32>
  }
  func.return %r : tensor<2x4xi32>
}"""


def test_matmul_result_becomes_a_zero_filled_buffer():
    """Tensor arguments and results become memrefs. The `dense<0>` result
    tensor becomes a `memref.alloc` plus a row-major fill nest. The buffer is
    returned, so it is not deallocated."""
    check_bufferization(MATMUL, """
    // CHECK: func.func @matmul(%{{.*}}: memref<2x3xi32>, %{{.*}}: memref<3x4xi32>) -> memref<2x4xi32>
    // CHECK: %[[BUF:.*]] = memref.alloc() : memref<2x4xi32>
    // CHECK-DAG: %[[ROWS:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[COLS:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: scf.for %[[FI:.*]] = %{{.*}} to %[[ROWS]] step %{{.*}} {
    // CHECK-NEXT: scf.for %[[FJ:.*]] = %{{.*}} to %[[COLS]] step %{{.*}} {
    // CHECK-NEXT: memref.store %[[ZERO]], %[[BUF]][%[[FI]], %[[FJ]]] : memref<2x4xi32>
    // CHECK-NOT: memref.dealloc
    // CHECK: func.return %[[BUF]] : memref<2x4xi32>
    """)


def test_matmul_nest_loads_and_stores_in_place():
    """Extracts become loads and the insert becomes a store into the result
    buffer. The i and j loops no longer carry the tensor (no iter_args),
    while the k loop keeps its i32 accumulator."""
    check_bufferization(MATMUL, """
    // CHECK: func.func @matmul(%[[A:.*]]: memref<2x3xi32>, %[[B:.*]]: memref<3x4xi32>)
    // CHECK: %[[BUF:.*]] = memref.alloc() : memref<2x4xi32>
    // CHECK: scf.for %[[I:.*]] = %c0 to %m step %c1 {
    // CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %n step %c1 {
    // CHECK: %[[SUM:.*]] = scf.for %[[K:.*]] = %c0 to %kd step %c1 iter_args(%[[ACC:.*]] = %zero) -> (i32) {
    // CHECK-NEXT: %[[X:.*]] = memref.load %[[A]][%[[I]], %[[K]]] : memref<2x3xi32>
    // CHECK-NEXT: %[[Y:.*]] = memref.load %[[B]][%[[K]], %[[J]]] : memref<3x4xi32>
    // CHECK: scf.yield
    // CHECK-NEXT: }
    // CHECK-NEXT: memref.store %[[SUM]], %[[BUF]][%[[I]], %[[J]]] : memref<2x4xi32>
    // CHECK: func.return %[[BUF]] : memref<2x4xi32>
    """)


def test_temporary_buffer_is_deallocated():
    """Both constants get a buffer from this pass. `%tmp` is only read, so its
    buffer is freed before the return. `%out` is returned, so the caller owns
    it."""
    check_bufferization("""
    func.func @f() -> tensor<2x2xi32> {
      %c0 = arith.constant 0 : index
      %tmp = arith.constant dense<7> : tensor<2x2xi32>
      %out = arith.constant dense<0> : tensor<2x2xi32>
      %x = tensor.extract %tmp[%c0, %c0] : tensor<2x2xi32>
      %r = tensor.insert %x into %out[%c0, %c0] : tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    }""", """
    // CHECK: %[[TMP:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK: %[[OUT:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK: %[[X:.*]] = memref.load %[[TMP]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: memref.store %[[X]], %[[OUT]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: memref.dealloc %[[TMP]] : memref<2x2xi32>
    // CHECK-NEXT: func.return %[[OUT]] : memref<2x2xi32>
    """)


def test_weight_constant_becomes_a_read_only_global():
    """A constant with different values (a weight) becomes a module-level
    `memref.global constant`, read through `memref.get_global`. The data
    lives in the executable, so the buffer is never deallocated. xDSL prints
    `memref.global` in generic form."""
    check_bufferization("""
    func.func @f() -> i32 {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %w = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
      %x = tensor.extract %w[%c0, %c1] : tensor<2x2xi32>
      func.return %x : i32
    }""", """
    // CHECK: "memref.global"() <{sym_name = "__constant_0", type = memref<2x2xi32>, initial_value = dense<{{\\[\\[}}1, 2], [3, 4]]> : tensor<2x2xi32>, sym_visibility = "private", constant}>
    // CHECK: func.func @f() -> i32
    // CHECK: %[[W:.*]] = memref.get_global @__constant_0 : memref<2x2xi32>
    // CHECK-NEXT: %[[X:.*]] = memref.load %[[W]][%c0, %c1] : memref<2x2xi32>
    // CHECK-NOT: memref.dealloc
    // CHECK: func.return %[[X]] : i32
    """)


def test_refuses_insert_into_a_weight_constant():
    """A store into the global would write to the executable's read-only
    data, so the pass refuses, as it does for a function argument."""
    with pytest.raises(NotImplementedError, match="not owned"):
        bufferize("""
        func.func @f() -> i32 {
          %c0 = arith.constant 0 : index
          %v = arith.constant 9 : i32
          %w = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
          %w2 = tensor.insert %v into %w[%c0, %c0] : tensor<2x2xi32>
          %x = tensor.extract %w2[%c0, %c0] : tensor<2x2xi32>
          func.return %x : i32
        }""")


def test_returned_argument_is_copied():
    """The caller frees the returned buffer, and it also frees its own input.
    Returning the input buffer itself would free it twice, so the pass
    returns a fresh copy that the caller owns."""
    check_bufferization("""
    func.func @f(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
      func.return %a : tensor<2x2xi32>
    }""", """
    // CHECK: func.func @f(%[[A:.*]]: memref<2x2xi32>) -> memref<2x2xi32>
    // CHECK-NEXT: %[[COPY:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: "memref.copy"(%[[A]], %[[COPY]]) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
    // CHECK-NEXT: func.return %[[COPY]] : memref<2x2xi32>
    """)


def test_scalar_argument_keeps_its_type():
    """Only tensor types become memrefs. The i32 argument and result pass
    through unchanged, and `%s` still feeds the `arith.addi`."""
    check_bufferization("""
    func.func @f(%t: tensor<2x2xi32>, %s: i32) -> i32 {
      %c0 = arith.constant 0 : index
      %x = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
      %y = arith.addi %x, %s : i32
      func.return %y : i32
    }""", """
    // CHECK: func.func @f(%[[T:.*]]: memref<2x2xi32>, %[[S:.*]]: i32) -> i32
    // CHECK: %[[X:.*]] = memref.load %[[T]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[Y:.*]] = arith.addi %[[X]], %[[S]] : i32
    // CHECK-NEXT: func.return %[[Y]] : i32
    """)


def test_function_without_tensors_is_left_untouched():
    module = parse("""
    func.func @f(%a: i32, %b: i32) -> i32 {
      %c = arith.addi %a, %b : i32
      func.return %c : i32
    }""")
    before = str(module)
    apply_bufferization(module)
    assert str(module) == before


def test_refuses_insert_into_a_tensor_with_a_second_use():
    """`%old` must still read 0 after the insert. A store into `%t`'s buffer
    would make it read 9, so the pass refuses instead of miscompiling."""
    with pytest.raises(NotImplementedError, match="used again elsewhere"):
        bufferize("""
        func.func @f() -> i32 {
          %c0 = arith.constant 0 : index
          %v = arith.constant 9 : i32
          %t = arith.constant dense<0> : tensor<2x2xi32>
          %t2 = tensor.insert %v into %t[%c0, %c0] : tensor<2x2xi32>
          %old = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
          func.return %old : i32
        }""")


def test_refuses_insert_into_a_function_argument():
    """A store into `%arg`'s buffer would change the caller's input."""
    with pytest.raises(NotImplementedError, match="function argument"):
        bufferize("""
        func.func @f(%arg: tensor<2x2xi32>) -> tensor<2x2xi32> {
          %c0 = arith.constant 0 : index
          %v = arith.constant 9 : i32
          %r = tensor.insert %v into %arg[%c0, %c0] : tensor<2x2xi32>
          func.return %r : tensor<2x2xi32>
        }""")
