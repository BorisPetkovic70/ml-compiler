"""FileCheck tests for vectorization: each test parses a small bufferized
function, runs only `apply_vectorization`, and matches the printed IR against
its `// CHECK:` lines.
"""
from conftest import parse, filecheck
from middle_end.vectorization import apply_vectorization


def vectorize(ir: str, width: int):
    module = parse(ir)
    apply_vectorization(module, width)
    module.verify()
    return module


def test_ikj_matmul_nest_is_vectorized():
    """The `%j` loop stores along a row of `%C`, so a vector loop takes its
    first 8 of 10 steps, 4 at a time. `%C` and `%B` are read with
    `vector.load`, and `%A[%i, %k]` is loaded as a scalar and broadcast
    where `arith.muli` uses it. The `%j` loop itself now starts at 8 and runs
    the last 2 steps."""
    filecheck(vectorize("""
    func.func @f(%A: memref<3x5xi32>, %B: memref<5x10xi32>, %C: memref<3x10xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 3 : index
      %N = arith.constant 10 : index
      %K = arith.constant 5 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %k = %c0 to %K step %c1 {
          scf.for %j = %c0 to %N step %c1 {
            %c = memref.load %C[%i, %j] : memref<3x10xi32>
            %a = memref.load %A[%i, %k] : memref<3x5xi32>
            %b = memref.load %B[%k, %j] : memref<5x10xi32>
            %p = arith.muli %a, %b : i32
            %s = arith.addi %c, %p : i32
            memref.store %s, %C[%i, %j] : memref<3x10xi32>
          }
        }
      }
      func.return
    }""", width=4), """
    // CHECK: scf.for %k = %c0 to %K step %c1 {
    // CHECK-NEXT: %[[W:.*]] = arith.constant 4 : index
    // CHECK-NEXT: %[[SPLIT:.*]] = arith.constant 8 : index
    // CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %[[SPLIT]] step %[[W]] {
    // CHECK-NEXT: %[[CV:.*]] = vector.load %C[%i, %[[J]]] : memref<3x10xi32>, vector<4xi32>
    // CHECK-NEXT: %[[A:.*]] = memref.load %A[%i, %k] : memref<3x5xi32>
    // CHECK-NEXT: %[[BV:.*]] = vector.load %B[%k, %[[J]]] : memref<5x10xi32>, vector<4xi32>
    // CHECK-NEXT: %[[AV:.*]] = vector.broadcast %[[A]] : i32 to vector<4xi32>
    // CHECK-NEXT: %[[PV:.*]] = arith.muli %[[AV]], %[[BV]] : vector<4xi32>
    // CHECK-NEXT: %[[SV:.*]] = arith.addi %[[CV]], %[[PV]] : vector<4xi32>
    // CHECK-NEXT: vector.store %[[SV]], %C[%i, %[[J]]] : memref<3x10xi32>, vector<4xi32>
    // CHECK-NEXT: }
    // CHECK-NEXT: scf.for %j = %[[SPLIT]] to %N step %c1 {
    // CHECK-NEXT: %c = memref.load %C[%i, %j] : memref<3x10xi32>
    """)


def test_ijk_matmul_nest_is_left_unchanged():
    """The innermost loop is `%k`, and every step stores to the one element
    `%C[%i, %j]`. There is no row to store a vector to, so the nest gets no
    vector loop."""
    module = vectorize("""
    func.func @f(%A: memref<3x5xi32>, %B: memref<5x10xi32>, %C: memref<3x10xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 3 : index
      %N = arith.constant 10 : index
      %K = arith.constant 5 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %j = %c0 to %N step %c1 {
          scf.for %k = %c0 to %K step %c1 {
            %c = memref.load %C[%i, %j] : memref<3x10xi32>
            %a = memref.load %A[%i, %k] : memref<3x5xi32>
            %b = memref.load %B[%k, %j] : memref<5x10xi32>
            %p = arith.muli %a, %b : i32
            %s = arith.addi %c, %p : i32
            memref.store %s, %C[%i, %j] : memref<3x10xi32>
          }
        }
      }
      func.return
    }""", width=4)
    filecheck(module, """
    // CHECK: scf.for %j = %c0 to %N step %c1 {
    // CHECK-NEXT: scf.for %k = %c0 to %K step %c1 {
    // CHECK-NEXT: %c = memref.load %C[%i, %j] : memref<3x10xi32>
    """)
    filecheck(module, "// CHECK-NOT: vector.")


def test_elementwise_nest_with_a_scalar_operand_is_vectorized():
    """`%R[%i, %j] = %A[%i, %j] + %x` stores along a row, so it is
    vectorized. `%x` comes from outside the loop and is broadcast. 8 is a
    multiple of 4, so no scalar loop is left."""
    module = vectorize("""
    func.func @f(%A: memref<2x8xi32>, %x: i32, %R: memref<2x8xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 2 : index
      %N = arith.constant 8 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %j = %c0 to %N step %c1 {
          %a = memref.load %A[%i, %j] : memref<2x8xi32>
          %s = arith.addi %a, %x : i32
          memref.store %s, %R[%i, %j] : memref<2x8xi32>
        }
      }
      func.return
    }""", width=4)
    filecheck(module, """
    // CHECK: scf.for %i = %c0 to %M step %c1 {
    // CHECK-NEXT: %[[W:.*]] = arith.constant 4 : index
    // CHECK-NEXT: %[[SPLIT:.*]] = arith.constant 8 : index
    // CHECK-NEXT: scf.for %[[J:.*]] = %c0 to %[[SPLIT]] step %[[W]] {
    // CHECK-NEXT: %[[AV:.*]] = vector.load %A[%i, %[[J]]] : memref<2x8xi32>, vector<4xi32>
    // CHECK-NEXT: %[[XV:.*]] = vector.broadcast %x : i32 to vector<4xi32>
    // CHECK-NEXT: %[[SV:.*]] = arith.addi %[[AV]], %[[XV]] : vector<4xi32>
    // CHECK-NEXT: vector.store %[[SV]], %R[%i, %[[J]]] : memref<2x8xi32>, vector<4xi32>
    // CHECK-NEXT: }
    // CHECK-NEXT: }
    // CHECK-NEXT: func.return
    """)


def test_max_nest_is_left_unchanged():
    """`hc.max` lowers to `arith.cmpi` and `scf.if`, which the pass does not
    rewrite, so the loop stays scalar."""
    module = vectorize("""
    func.func @f(%A: memref<2x8xi32>, %x: i32, %R: memref<2x8xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 2 : index
      %N = arith.constant 8 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %j = %c0 to %N step %c1 {
          %a = memref.load %A[%i, %j] : memref<2x8xi32>
          %gt = arith.cmpi sgt, %a, %x : i32
          %m = scf.if %gt -> (i32) {
            scf.yield %a : i32
          } else {
            scf.yield %x : i32
          }
          memref.store %m, %R[%i, %j] : memref<2x8xi32>
        }
      }
      func.return
    }""", width=4)
    filecheck(module, """
    // CHECK: scf.for %i = %c0 to %M step %c1 {
    // CHECK-NEXT: scf.for %j = %c0 to %N step %c1 {
    // CHECK-NEXT: %a = memref.load %A[%i, %j] : memref<2x8xi32>
    """)
    filecheck(module, "// CHECK-NOT: vector.")
