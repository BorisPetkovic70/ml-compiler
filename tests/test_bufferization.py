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
  %zero = arith.constant 0 : i32
  %init = tensor.empty() : tensor<2x4xi32>
  %zeroed = scf.for %fi = %c0 to %m step %c1 iter_args(%zi = %init) -> (tensor<2x4xi32>) {
    %zr = scf.for %fj = %c0 to %n step %c1 iter_args(%zj = %zi) -> (tensor<2x4xi32>) {
      %z = tensor.insert %zero into %zj[%fi, %fj] : tensor<2x4xi32>
      scf.yield %z : tensor<2x4xi32>
    }
    scf.yield %zr : tensor<2x4xi32>
  }
  %r = scf.for %i = %c0 to %m step %c1 iter_args(%ti = %zeroed) -> (tensor<2x4xi32>) {
    %rj = scf.for %j = %c0 to %n step %c1 iter_args(%tj = %ti) -> (tensor<2x4xi32>) {
      %rk = scf.for %k = %c0 to %kd step %c1 iter_args(%tk = %tj) -> (tensor<2x4xi32>) {
        %c = tensor.extract %tk[%i, %j] : tensor<2x4xi32>
        %x = tensor.extract %a[%i, %k] : tensor<2x3xi32>
        %y = tensor.extract %b[%k, %j] : tensor<3x4xi32>
        %p = arith.muli %x, %y : i32
        %s = arith.addi %c, %p : i32
        %t = tensor.insert %s into %tk[%i, %j] : tensor<2x4xi32>
        scf.yield %t : tensor<2x4xi32>
      }
      scf.yield %rk : tensor<2x4xi32>
    }
    scf.yield %rj : tensor<2x4xi32>
  }
  func.return %r : tensor<2x4xi32>
}"""


def test_matmul_result_becomes_a_zero_filled_buffer():
    """Tensor arguments and results become memrefs. The `tensor.empty` result
    becomes a bare `memref.alloc`, and the fill nest stores a 0 into each of
    its elements. The buffer is returned, so it is not deallocated."""
    check_bufferization(MATMUL, """
    // CHECK: func.func @matmul(%{{.*}}: memref<2x3xi32>, %{{.*}}: memref<3x4xi32>) -> memref<2x4xi32>
    // CHECK: %[[BUF:.*]] = memref.alloc() : memref<2x4xi32>
    // CHECK-NEXT: scf.for %[[I:.*]] = %c0 to %m step %c1 {
    // CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %n step %c1 {
    // CHECK-NEXT: memref.store %zero, %[[BUF]][%[[I]], %[[J]]] : memref<2x4xi32>
    // CHECK-NOT: memref.dealloc
    // CHECK: func.return %[[BUF]] : memref<2x4xi32>
    """)


def test_matmul_nest_loads_and_stores_in_place():
    """Extracts become loads and the insert becomes a store. The nest reads
    C[i, j] and writes it back to the same buffer the fill nest wrote, with
    no copy. No loop carries the tensor any more (no iter_args), and each
    body holds only the next loop."""
    check_bufferization(MATMUL, """
    // CHECK: func.func @matmul(%[[A:.*]]: memref<2x3xi32>, %[[B:.*]]: memref<3x4xi32>)
    // CHECK: %[[BUF:.*]] = memref.alloc() : memref<2x4xi32>
    // CHECK-NOT: memref.alloc
    // CHECK: memref.store %zero
    // CHECK: scf.for %[[I:.*]] = %c0 to %m step %c1 {
    // CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %n step %c1 {
    // CHECK-NEXT: scf.for %[[K:.*]] = %c0 to %kd step %c1 {
    // CHECK-NEXT: %[[C:.*]] = memref.load %[[BUF]][%[[I]], %[[J]]] : memref<2x4xi32>
    // CHECK-NEXT: %[[X:.*]] = memref.load %[[A]][%[[I]], %[[K]]] : memref<2x3xi32>
    // CHECK-NEXT: %[[Y:.*]] = memref.load %[[B]][%[[K]], %[[J]]] : memref<3x4xi32>
    // CHECK-NEXT: %[[P:.*]] = arith.muli %[[X]], %[[Y]] : i32
    // CHECK-NEXT: %[[S:.*]] = arith.addi %[[C]], %[[P]] : i32
    // CHECK-NEXT: memref.store %[[S]], %[[BUF]][%[[I]], %[[J]]] : memref<2x4xi32>
    // CHECK: func.return %[[BUF]] : memref<2x4xi32>
    """)


def test_temporary_buffer_is_deallocated():
    """Both `tensor.empty` ops get a buffer from this pass. `%tmp` is not
    returned, so its buffer is freed before the return. `%out` is returned,
    so the caller owns it."""
    check_bufferization("""
    func.func @f(%v: i32) -> tensor<2x2xi32> {
      %c0 = arith.constant 0 : index
      %e = tensor.empty() : tensor<2x2xi32>
      %tmp = tensor.insert %v into %e[%c0, %c0] : tensor<2x2xi32>
      %out = tensor.empty() : tensor<2x2xi32>
      %x = tensor.extract %tmp[%c0, %c0] : tensor<2x2xi32>
      %r = tensor.insert %x into %out[%c0, %c0] : tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    }""", """
    // CHECK: func.func @f(%[[V:.*]]: i32) -> memref<2x2xi32>
    // CHECK: %[[TMP:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: memref.store %[[V]], %[[TMP]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[OUT:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: %[[X:.*]] = memref.load %[[TMP]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: memref.store %[[X]], %[[OUT]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: memref.dealloc %[[TMP]] : memref<2x2xi32>
    // CHECK-NEXT: func.return %[[OUT]] : memref<2x2xi32>
    """)


def test_constant_becomes_a_read_only_global():
    """A tensor constant (a weight) becomes a module-level
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


def test_read_before_write_stores_in_place():
    """`%t` has two uses, but the extract runs before the insert and nothing
    reads `%t` afterwards. The insert is its last use, so the store goes
    into the same buffer."""
    check_bufferization("""
    func.func @f() -> i32 {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : i32
      %five = arith.constant 5 : i32
      %e = tensor.empty() : tensor<2x2xi32>
      %t = tensor.insert %five into %e[%c0, %c0] : tensor<2x2xi32>
      %old = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
      %inc = arith.addi %old, %one : i32
      %t2 = tensor.insert %inc into %t[%c0, %c0] : tensor<2x2xi32>
      %x = tensor.extract %t2[%c0, %c0] : tensor<2x2xi32>
      func.return %x : i32
    }""", """
    // CHECK: %[[BUF:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: memref.store %five, %[[BUF]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[OLD:.*]] = memref.load %[[BUF]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[INC:.*]] = arith.addi %[[OLD]], %one : i32
    // CHECK-NEXT: memref.store %[[INC]], %[[BUF]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[X:.*]] = memref.load %[[BUF]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: memref.dealloc %[[BUF]] : memref<2x2xi32>
    // CHECK-NEXT: func.return %[[X]] : i32
    """)


def test_insert_into_a_tensor_read_afterwards_writes_to_a_copy():
    """`%old` must still read 5 after the insert. A store into `%t`'s buffer
    would make it read 9, so the store goes into a copy and `%old` loads
    from the original. Both buffers are temporaries."""
    check_bufferization("""
    func.func @f() -> i32 {
      %c0 = arith.constant 0 : index
      %v = arith.constant 9 : i32
      %five = arith.constant 5 : i32
      %e = tensor.empty() : tensor<2x2xi32>
      %t = tensor.insert %five into %e[%c0, %c0] : tensor<2x2xi32>
      %t2 = tensor.insert %v into %t[%c0, %c0] : tensor<2x2xi32>
      %old = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
      func.return %old : i32
    }""", """
    // CHECK: %[[T:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: memref.store %five, %[[T]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[COPY:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: "memref.copy"(%[[T]], %[[COPY]]) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
    // CHECK-NEXT: memref.store %v, %[[COPY]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: %[[OLD:.*]] = memref.load %[[T]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: memref.dealloc %[[T]] : memref<2x2xi32>
    // CHECK-NEXT: memref.dealloc %[[COPY]] : memref<2x2xi32>
    // CHECK-NEXT: func.return %[[OLD]] : i32
    """)


def test_refuses_insert_in_a_loop_into_a_tensor_defined_outside_it():
    """The insert is `%t`'s only use, but it runs once per iteration, and
    each iteration must start from the original `%t`. A store would keep the
    previous iteration's write in the buffer. The fix would be a copy per
    iteration, which this pass does not place inside a loop body."""
    with pytest.raises(NotImplementedError, match="nested region"):
        bufferize("""
        func.func @f() -> i32 {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %v = arith.constant 9 : i32
          %z = arith.constant 0 : i32
          %t = tensor.empty() : tensor<2x2xi32>
          %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %z) -> (i32) {
            %t2 = tensor.insert %v into %t[%i, %c0] : tensor<2x2xi32>
            %x = tensor.extract %t2[%c0, %c0] : tensor<2x2xi32>
            %s = arith.addi %acc, %x : i32
            scf.yield %s : i32
          }
          func.return %r : i32
        }""")


def test_insert_into_a_constant_writes_to_a_copy():
    """A constant's data is read-only, so the store goes into a copy of the
    global. The function owns the copy, so it is returned as it is."""
    check_bufferization("""
    func.func @f(%v: i32) -> tensor<2x2xi32> {
      %c0 = arith.constant 0 : index
      %w = arith.constant dense<7> : tensor<2x2xi32>
      %r = tensor.insert %v into %w[%c0, %c0] : tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    }""", """
    // CHECK: func.func @f(%[[V:.*]]: i32) -> memref<2x2xi32>
    // CHECK: %[[W:.*]] = memref.get_global @__constant_0 : memref<2x2xi32>
    // CHECK-NEXT: %[[COPY:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: "memref.copy"(%[[W]], %[[COPY]]) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
    // CHECK-NEXT: memref.store %[[V]], %[[COPY]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: func.return %[[COPY]] : memref<2x2xi32>
    """)


def test_insert_into_a_function_argument_writes_to_a_copy():
    """A store into `%arg`'s buffer would change the caller's input, so the
    store goes into a copy. The function owns the copy, so it is returned
    as it is."""
    check_bufferization("""
    func.func @f(%arg: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %c0 = arith.constant 0 : index
      %v = arith.constant 9 : i32
      %r = tensor.insert %v into %arg[%c0, %c0] : tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    }""", """
    // CHECK: func.func @f(%[[ARG:.*]]: memref<2x2xi32>) -> memref<2x2xi32>
    // CHECK: %[[COPY:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: "memref.copy"(%[[ARG]], %[[COPY]]) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
    // CHECK-NEXT: memref.store %v, %[[COPY]][%c0, %c0] : memref<2x2xi32>
    // CHECK-NEXT: func.return %[[COPY]] : memref<2x2xi32>
    """)


def test_loop_carrying_a_function_argument_writes_to_a_copy():
    """The loop's stores would land in `%a`'s buffer, so `%a` is copied once
    before the loop and the loop writes into the copy."""
    check_bufferization("""
    func.func @f(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %v = arith.constant 9 : i32
      %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%t = %a) -> (tensor<2x2xi32>) {
        %t2 = tensor.insert %v into %t[%i, %c0] : tensor<2x2xi32>
        scf.yield %t2 : tensor<2x2xi32>
      }
      func.return %r : tensor<2x2xi32>
    }""", """
    // CHECK: func.func @f(%[[A:.*]]: memref<2x2xi32>) -> memref<2x2xi32>
    // CHECK: %[[COPY:.*]] = memref.alloc() : memref<2x2xi32>
    // CHECK-NEXT: "memref.copy"(%[[A]], %[[COPY]]) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
    // CHECK-NEXT: scf.for %[[I:.*]] = %c0 to %c2 step %c1 {
    // CHECK-NEXT: memref.store %v, %[[COPY]][%[[I]], %c0] : memref<2x2xi32>
    // CHECK: func.return %[[COPY]] : memref<2x2xi32>
    """)
