"""Tests of the interpreter itself, the oracle the other tests trust. Each
parses a small function and checks the value the interpreter returns.
"""
import pytest

from conftest import parse, run, lower


def test_tensor_insert_has_value_semantics():
    """`tensor.insert` returns a new tensor and leaves `%t` unchanged, so
    `%old` still reads 2, not 9."""
    module = parse("""
    func.func @f() -> i32 {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %t = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
      %nine = arith.constant 9 : i32
      %t2 = tensor.insert %nine into %t[%c0, %c1] : tensor<2x2xi32>
      %old = tensor.extract %t[%c0, %c1] : tensor<2x2xi32>
      func.return %old : i32
    }""")
    assert run(module, [], "f") == 2


def test_scf_for_threads_iter_args():
    """Fibonacci: each iteration yields `(b, a + b)` as the next `(a, b)`.
    Mixing up the order of the iter_args, the yielded values or the loop's
    results changes the answer."""
    module = parse("""
    func.func @f() -> i32 {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 6 : index
      %a0 = arith.constant 0 : i32
      %b0 = arith.constant 1 : i32
      %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%a = %a0, %b = %b0) -> (i32, i32) {
        %s = arith.addi %a, %b : i32
        scf.yield %b, %s : i32, i32
      }
      func.return %r#0 : i32
    }""")
    assert run(module, [], "f") == 8  # fib(6)


@pytest.mark.parametrize("row,col", [(2, 0), (0, -1)])
def test_tensor_extract_out_of_bounds_raises(row, col):
    """A Python list would silently wrap the negative index."""
    module = parse(f"""
    func.func @f() -> i32 {{
      %row = arith.constant {row} : index
      %col = arith.constant {col} : index
      %t = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
      %x = tensor.extract %t[%row, %col] : tensor<2x2xi32>
      func.return %x : i32
    }}""")
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(module, [], "f")


def test_pow_with_exponent_zero():
    """x ** 0 is 1. After lowering, the multiply loop runs zero times and
    returns its initial accumulator."""
    module = parse("""
    func.func @f(%x: i32, %e: i32) -> i32 {
      %r = "hc.pow"(%x, %e) : (i32, i32) -> i32
      func.return %r : i32
    }""")
    assert run(module, [7, 0], "f") == 1
    lower(module)
    assert run(module, [7, 0], "f") == 1


def test_memref_store_writes_through_to_the_callers_buffer():
    """Unlike `tensor.insert`, `memref.store` changes the buffer in place.
    `%buf` aliases the caller's buffer, so the caller sees the 7 too. This
    is why bufferization refuses to store into a function argument."""
    buf = [[1, 2], [3, 4]]
    module = parse("""
    func.func @f(%buf: memref<2x2xi32>) -> i32 {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %seven = arith.constant 7 : i32
      memref.store %seven, %buf[%c0, %c1] : memref<2x2xi32>
      %x = memref.load %buf[%c0, %c1] : memref<2x2xi32>
      func.return %x : i32
    }""")
    assert run(module, [buf], "f") == 7
    assert buf == [[1, 7], [3, 4]]


@pytest.mark.parametrize("access", [
    "%y = memref.load %buf[%c2, %c0] : memref<2x2xi32>",
    "memref.store %x, %buf[%c0, %neg] : memref<2x2xi32>",
])
def test_memref_access_out_of_bounds_raises(access):
    """Both a load past the last row and a store at a negative column."""
    module = parse(f"""
    func.func @f(%buf: memref<2x2xi32>, %x: i32) -> i32 {{
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %neg = arith.constant -1 : index
      {access}
      func.return %x : i32
    }}""")
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(module, [[[1, 2], [3, 4]], 7], "f")


def test_memref_use_after_dealloc_raises():
    """`memref.dealloc` drops the buffer, so a later load can't find it."""
    module = parse("""
    func.func @f() -> i32 {
      %c0 = arith.constant 0 : index
      %buf = memref.alloc() : memref<2x2xi32>
      memref.dealloc %buf : memref<2x2xi32>
      %x = memref.load %buf[%c0, %c0] : memref<2x2xi32>
      func.return %x : i32
    }""")
    with pytest.raises(KeyError):
        run(module, [], "f")


def test_vector_store_and_load_move_a_run_of_elements():
    """`vector.load` reads 4 elements of row 1, starting at column 1.
    `vector.store` writes them to row 0 from column 2 on, in place."""
    buf = [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]]
    module = parse("""
    func.func @f(%buf: memref<2x6xi32>) -> vector<4xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %v = vector.load %buf[%c1, %c1] : memref<2x6xi32>, vector<4xi32>
      vector.store %v, %buf[%c0, %c2] : memref<2x6xi32>, vector<4xi32>
      func.return %v : vector<4xi32>
    }""")
    assert run(module, [buf], "f") == [2, 3, 4, 5]
    assert buf == [[0, 0, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]


def test_vector_load_past_the_end_of_a_row_raises():
    """A 4-element load from column 3 of a 6-element row needs column 6."""
    module = parse("""
    func.func @f(%buf: memref<2x6xi32>) -> vector<4xi32> {
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %v = vector.load %buf[%c0, %c3] : memref<2x6xi32>, vector<4xi32>
      func.return %v : vector<4xi32>
    }""")
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(module, [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]], "f")
