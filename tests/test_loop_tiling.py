"""FileCheck tests for loop tiling: each test parses a small bufferized
function, runs only `apply_loop_tiling`, and matches the printed IR against
its `// CHECK:` lines.
"""
from conftest import parse, filecheck
from middle_end.loop_tiling import apply_loop_tiling


def tile(ir: str, tile_size: int):
    module = parse(ir)
    apply_loop_tiling(module, tile_size)
    module.verify()
    return module


def test_matmul_nest_is_tiled():
    """The `%i` and `%k` loops each get a tile loop that steps `%T` at a
    time. They become the point loops: `%i` runs from its tile's start
    `%II` to `min(%II + %T, %M)`, and `%k` likewise. The `%j` loop and the
    body are unchanged."""
    filecheck(tile("""
    func.func @f(%A: memref<3x5xi32>, %B: memref<5x4xi32>, %C: memref<3x4xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %M = arith.constant 3 : index
      %N = arith.constant 4 : index
      %K = arith.constant 5 : index
      scf.for %i = %c0 to %M step %c1 {
        scf.for %k = %c0 to %K step %c1 {
          scf.for %j = %c0 to %N step %c1 {
            %c = memref.load %C[%i, %j] : memref<3x4xi32>
            %a = memref.load %A[%i, %k] : memref<3x5xi32>
            %b = memref.load %B[%k, %j] : memref<5x4xi32>
            %p = arith.muli %a, %b : i32
            %s = arith.addi %c, %p : i32
            memref.store %s, %C[%i, %j] : memref<3x4xi32>
          }
        }
      }
      func.return
    }""", tile_size=2), """
    // CHECK: %[[T:.*]] = arith.constant 2 : index
    // CHECK-NEXT: scf.for %[[II:.*]] = %c0 to %M step %[[T]] {
    // CHECK-NEXT: scf.for %[[KK:.*]] = %c0 to %K step %[[T]] {
    // CHECK-NEXT: %[[I_END:.*]] = arith.addi %[[II]], %[[T]] : index
    // CHECK-NEXT: %[[I_STOP:.*]] = arith.minsi %[[I_END]], %M : index
    // CHECK-NEXT: %[[K_END:.*]] = arith.addi %[[KK]], %[[T]] : index
    // CHECK-NEXT: %[[K_STOP:.*]] = arith.minsi %[[K_END]], %K : index
    // CHECK-NEXT: scf.for %i = %[[II]] to %[[I_STOP]] step %c1 {
    // CHECK-NEXT: scf.for %k = %[[KK]] to %[[K_STOP]] step %c1 {
    // CHECK-NEXT: scf.for %j = %c0 to %N step %c1 {
    // CHECK-NEXT: %c = memref.load %C[%i, %j] : memref<3x4xi32>
    // CHECK-NEXT: %a = memref.load %A[%i, %k] : memref<3x5xi32>
    // CHECK-NEXT: %b = memref.load %B[%k, %j] : memref<5x4xi32>
    // CHECK-NEXT: %p = arith.muli %a, %b : i32
    // CHECK-NEXT: %s = arith.addi %c, %p : i32
    // CHECK-NEXT: memref.store %s, %C[%i, %j] : memref<3x4xi32>
    """)


def test_elementwise_nest_is_left_unchanged():
    """The body stores to `%R` and never loads it, so it is not an in-place
    update and the nest gets no tile loops."""
    filecheck(tile("""
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
    }""", tile_size=2), """
    // CHECK: %N = arith.constant 3 : index
    // CHECK-NEXT: scf.for %i = %c0 to %M step %c1 {
    // CHECK-NEXT: scf.for %j = %c0 to %N step %c1 {
    // CHECK-NEXT: %a = memref.load %A[%i, %j] : memref<2x3xi32>
    """)
