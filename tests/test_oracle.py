"""Oracle tests: the interpreter runs an `hc` function, then the whole
middle end runs, then the interpreter runs the result again. Both runs must
give the expected value, computed by hand or by NumPy.

The inputs are function arguments, so constant folding can't replace the
lowered code with a precomputed result.
"""
import numpy as np
import pytest

from conftest import parse, run
from middle_end.pipeline import MiddleEndPipeline, MiddleEndPipelineConfig


def check_oracle(ir: str, args: list, expected, config=MiddleEndPipelineConfig()) -> None:
    """Checks `@f(*args) == expected` before and after the middle end. The
    default `config` runs lowering, bufferization, constant folding, constant
    CSE and DCE."""
    module = parse(ir)
    assert run(module, args, "f") == expected
    MiddleEndPipeline(config).apply_passes(module)
    module.verify()
    assert run(module, args, "f") == expected


def tensor_type(shape) -> str:
    return f"tensor<{'x'.join(str(d) for d in shape)}xi32>"


def value_type(shape) -> str:
    """`i32` for the scalar shape `()`, otherwise the tensor type."""
    return tensor_type(shape) if shape else "i32"


@pytest.mark.parametrize("op,ty,a,b,expected", [
    ("hc.add", "i32", 3, 4, 7),
    ("hc.sub", "i32", 10, 4, 6),
    ("hc.mul", "i32", 3, 4, 12),
    ("hc.max", "i32", 3, 9, 9),
    ("hc.min", "i32", 3, 9, 3),
    ("hc.pow", "i32", -2, 3, -8),
    ("hc.add", "tensor<2x2xi32>", [[1, 2], [3, 4]], [[10, 20], [30, 40]], [[11, 22], [33, 44]]),
    ("hc.sub", "tensor<2x2xi32>", [[10, 20], [30, 40]], [[1, 2], [3, 4]], [[9, 18], [27, 36]]),
    ("hc.mul", "tensor<2x2xi32>", [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[2, 6], [12, 20]]),
    # rank 1: a single loop
    ("hc.add", "tensor<4xi32>", [1, 2, 3, 4], [10, 20, 30, 40], [11, 22, 33, 44]),
    # batched: the lowering wraps the row/column nest in a loop over the leading dim
    ("hc.add", "tensor<2x2x2xi32>",
     [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
     [[[10, 20], [30, 40]], [[50, 60], [70, 80]]],
     [[[11, 22], [33, 44]], [[55, 66], [77, 88]]]),
])
def test_binary_op(op, ty, a, b, expected):
    check_oracle(f"""
    func.func @f(%a: {ty}, %b: {ty}) -> {ty} {{
      %r = "{op}"(%a, %b) : ({ty}, {ty}) -> {ty}
      func.return %r : {ty}
    }}""", [a, b], expected)


@pytest.mark.parametrize("a_shape,b_shape", [
    ((2, 3), (3,)),    # aligned at the last dim
    ((2, 3), (1, 3)),  # a size-1 dim is stretched
    ((2, 1), (1, 3)),  # each operand stretches one dim
    ((2, 3), ()),      # tensor - scalar
    ((), (2, 3)),      # scalar - tensor
    ((2, 3, 2, 2), (2, 2)),  # rank 4: one loop per dim
])
def test_broadcast_matches_numpy(a_shape, b_shape):
    """Subtraction, so swapped operands give a different result."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    r = a - b
    a_ty, b_ty, r_ty = value_type(a_shape), value_type(b_shape), tensor_type(r.shape)
    check_oracle(f"""
    func.func @f(%a: {a_ty}, %b: {b_ty}) -> {r_ty} {{
      %r = "hc.sub"(%a, %b) : ({a_ty}, {b_ty}) -> {r_ty}
      func.return %r : {r_ty}
    }}""", [a.tolist(), b.tolist()], r.tolist())


@pytest.mark.parametrize("op,reference", [
    ("hc.max", np.maximum), ("hc.min", np.minimum), ("hc.pow", np.power),
])
def test_max_min_pow_on_tensors_match_numpy(op, reference):
    """2x3 with 3, so the operands also broadcast. `%b` is never negative:
    `hc.pow` is defined for an exponent >= 0."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=(2, 3))
    b = rng.integers(0, 4, size=(3,))
    check_oracle(f"""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<3xi32>) -> tensor<2x3xi32> {{
      %r = "{op}"(%a, %b) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }}""", [a.tolist(), b.tolist()], reference(a, b).tolist())


@pytest.mark.parametrize("op,ty,x,expected", [
    ("hc.relu", "i32", -3, 0),
    ("hc.relu", "tensor<2x2xi32>", [[-1, 2], [3, -4]], [[0, 2], [3, 0]]),
])
def test_relu(op, ty, x, expected):
    check_oracle(f"""
    func.func @f(%x: {ty}) -> {ty} {{
      %r = "{op}"(%x) : ({ty}) -> {ty}
      func.return %r : {ty}
    }}""", [x], expected)


@pytest.mark.parametrize("a_shape,b_shape", [
    ((2, 3), (3, 4)),
    ((1, 5), (5, 1)),
    ((3, 1), (1, 2)),
    ((2, 2, 3), (2, 3, 4)),  # batched: one matmul per leading index
    ((2, 3, 2, 3), (2, 3, 3, 4)),  # two batch dims
])
def test_matmul_matches_numpy(a_shape, b_shape):
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    r = a @ b
    a_ty, b_ty, r_ty = tensor_type(a.shape), tensor_type(b.shape), tensor_type(r.shape)
    check_oracle(f"""
    func.func @f(%a: {a_ty}, %b: {b_ty}) -> {r_ty} {{
      %r = "hc.matmul"(%a, %b) : ({a_ty}, {b_ty}) -> {r_ty}
      func.return %r : {r_ty}
    }}""", [a.tolist(), b.tolist()], r.tolist())


@pytest.mark.parametrize("a_shape,b_shape", [
    ((2, 3), (3, 4)),
    ((2, 2, 3), (2, 3, 4)),  # batched: the batch loop stays outermost
])
def test_matmul_with_loop_interchange_matches_numpy(a_shape, b_shape):
    """The matmul nest runs in `i, k, j` order. Each `C[i, j]` still gets the
    same products added, so the result is the same."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    r = a @ b
    a_ty, b_ty, r_ty = tensor_type(a.shape), tensor_type(b.shape), tensor_type(r.shape)
    check_oracle(f"""
    func.func @f(%a: {a_ty}, %b: {b_ty}) -> {r_ty} {{
      %r = "hc.matmul"(%a, %b) : ({a_ty}, {b_ty}) -> {r_ty}
      func.return %r : {r_ty}
    }}""", [a.tolist(), b.tolist()], r.tolist(),
        config=MiddleEndPipelineConfig(interchange_loops=True))


@pytest.mark.parametrize("a_shape,b_shape", [
    ((3, 5), (5, 4)),  # 3 and 5 are not multiples of the tile size
    ((2, 3, 5), (2, 5, 4)),  # batched: the batch loop is not tiled
])
def test_matmul_with_loop_tiling_matches_numpy(a_shape, b_shape):
    """The `i, k, j` nest runs in 2 x 2 tiles of `i` and `k`. The last tile
    of each is cut short at the loop's bound, so every `(i, k)` pair still
    runs exactly once."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    r = a @ b
    a_ty, b_ty, r_ty = tensor_type(a.shape), tensor_type(b.shape), tensor_type(r.shape)
    check_oracle(f"""
    func.func @f(%a: {a_ty}, %b: {b_ty}) -> {r_ty} {{
      %r = "hc.matmul"(%a, %b) : ({a_ty}, {b_ty}) -> {r_ty}
      func.return %r : {r_ty}
    }}""", [a.tolist(), b.tolist()], r.tolist(),
        config=MiddleEndPipelineConfig(interchange_loops=True, tile_size=2))


@pytest.mark.parametrize("a_shape,b_shape", [
    ((3, 5), (5, 10)),  # 10 columns: two vectors, then 2 scalar steps
    ((3, 5), (5, 8)),  # 8 columns: two vectors and no scalar loop
    ((2, 3, 5), (2, 5, 10)),  # batched
])
def test_matmul_with_vectorization_matches_numpy(a_shape, b_shape):
    """The `i, k, j` nest, tiled, runs its `j` loop 4 columns at a time.
    Each `C[i, j]` still gets the same products added, whether a vector step
    or a scalar step adds them."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    r = a @ b
    a_ty, b_ty, r_ty = tensor_type(a.shape), tensor_type(b.shape), tensor_type(r.shape)
    check_oracle(f"""
    func.func @f(%a: {a_ty}, %b: {b_ty}) -> {r_ty} {{
      %r = "hc.matmul"(%a, %b) : ({a_ty}, {b_ty}) -> {r_ty}
      func.return %r : {r_ty}
    }}""", [a.tolist(), b.tolist()], r.tolist(),
        config=MiddleEndPipelineConfig(interchange_loops=True, tile_size=2, vector_width=4))


@pytest.mark.parametrize("op,reference", [
    ("hc.add", np.add),
    ("hc.mul", np.multiply),
    ("hc.max", np.maximum),  # lowers to scf.if: its loop stays scalar
])
def test_elementwise_with_vectorization_matches_numpy(op, reference):
    """`relu(op(a, b))` on 2x10 tensors, 4 columns at a time. `b` has shape
    2x1, so each row's single element is broadcast into a vector."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=(2, 10))
    b = rng.integers(-5, 6, size=(2, 1))
    check_oracle(f"""
    func.func @f(%a: tensor<2x10xi32>, %b: tensor<2x1xi32>) -> tensor<2x10xi32> {{
      %t = "{op}"(%a, %b) : (tensor<2x10xi32>, tensor<2x1xi32>) -> tensor<2x10xi32>
      %r = "hc.relu"(%t) : (tensor<2x10xi32>) -> tensor<2x10xi32>
      func.return %r : tensor<2x10xi32>
    }}""", [a.tolist(), b.tolist()], np.maximum(reference(a, b), 0).tolist(),
        config=MiddleEndPipelineConfig(vector_width=4))


def test_matmul_with_weight_constant_matches_numpy():
    """`%w` is a weight, as the loader builds it from an ONNX initializer.
    After bufferization it is read from a `memref.global`."""
    x = np.array([[1, 2], [3, 4]])
    w = np.array([[1, 2, 3], [4, 5, 6]])
    check_oracle("""
    func.func @f(%x: tensor<2x2xi32>) -> tensor<2x3xi32> {
      %w = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
      %r = "hc.matmul"(%x, %w) : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
      func.return %r : tensor<2x3xi32>
    }""", [x.tolist()], (x @ w).tolist())


def test_returned_input_is_a_copy():
    """After bufferization the function returns a copy of its input buffer
    (`memref.copy`), with the same values."""
    check_oracle("""
    func.func @f(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
      func.return %a : tensor<2x2xi32>
    }""", [[[1, 2], [3, 4]]], [[1, 2], [3, 4]])


def test_matmul_then_relu_matches_numpy():
    """The matmul's result feeds the relu's loop nest. After bufferization
    it is a temporary buffer."""
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=(2, 3))
    b = rng.integers(-5, 6, size=(3, 4))
    check_oracle("""
    func.func @f(%a: tensor<2x3xi32>, %b: tensor<3x4xi32>) -> tensor<2x4xi32> {
      %m = "hc.matmul"(%a, %b) : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
      %r = "hc.relu"(%m) : (tensor<2x4xi32>) -> tensor<2x4xi32>
      func.return %r : tensor<2x4xi32>
    }""", [a.tolist(), b.tolist()], np.maximum(a @ b, 0).tolist())


def test_read_before_write_in_place():
    """`%t` is read, then written. Bufferization stores into the same
    buffer, and the result is unchanged."""
    check_oracle("""
    func.func @f(%d: i32) -> i32 {
      %c0 = arith.constant 0 : index
      %five = arith.constant 5 : i32
      %e = tensor.empty() : tensor<2x2xi32>
      %t = tensor.insert %five into %e[%c0, %c0] : tensor<2x2xi32>
      %old = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
      %inc = arith.addi %old, %d : i32
      %t2 = tensor.insert %inc into %t[%c0, %c0] : tensor<2x2xi32>
      %x = tensor.extract %t2[%c0, %c0] : tensor<2x2xi32>
      func.return %x : i32
    }""", [3], 8)


def test_insert_into_an_input_still_read_afterwards():
    """`%a` is the caller's buffer and is read again after the insert.
    Bufferization stores into a copy, so `%old` still reads the input."""
    check_oracle("""
    func.func @f(%a: tensor<2x2xi32>, %v: i32) -> i32 {
      %c0 = arith.constant 0 : index
      %t = tensor.insert %v into %a[%c0, %c0] : tensor<2x2xi32>
      %old = tensor.extract %a[%c0, %c0] : tensor<2x2xi32>
      %new = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
      %d = arith.subi %new, %old : i32
      func.return %d : i32
    }""", [[[1, 2], [3, 4]], 9], 8)
