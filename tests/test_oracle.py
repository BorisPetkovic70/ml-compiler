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


def check_oracle(ir: str, args: list, expected) -> None:
    """Checks `@f(*args) == expected` before and after the middle end
    (lowering, bufferization, constant folding and DCE)."""
    module = parse(ir)
    assert run(module, args, "f") == expected
    MiddleEndPipeline(MiddleEndPipelineConfig()).apply_passes(module)
    module.verify()
    assert run(module, args, "f") == expected


def tensor_type(shape) -> str:
    return f"tensor<{'x'.join(str(d) for d in shape)}xi32>"


@pytest.mark.parametrize("op,ty,a,b,expected", [
    ("hc.add", "i32", 3, 4, 7),
    ("hc.sub", "i32", 10, 4, 6),
    ("hc.mul", "i32", 3, 4, 12),
    ("hc.max", "i32", 3, 9, 9),
    ("hc.min", "i32", 3, 9, 3),
    ("hc.pow", "i32", -2, 3, -8),
    ("hc.add_vec", "vector<4xi32>", [1, 2, 3, 4], [10, 20, 30, 40], [11, 22, 33, 44]),
    ("hc.sub_vec", "vector<4xi32>", [10, 20, 30, 40], [1, 2, 3, 4], [9, 18, 27, 36]),
    ("hc.mul_vec_vec", "vector<4xi32>", [1, 2, 3, 4], [2, 3, 4, 5], [2, 6, 12, 20]),
    ("hc.add_tensor", "tensor<2x2xi32>", [[1, 2], [3, 4]], [[10, 20], [30, 40]], [[11, 22], [33, 44]]),
    ("hc.sub_tensor", "tensor<2x2xi32>", [[10, 20], [30, 40]], [[1, 2], [3, 4]], [[9, 18], [27, 36]]),
    ("hc.mul_tensor", "tensor<2x2xi32>", [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[2, 6], [12, 20]]),
    # batched: the lowering wraps the row/column nest in a loop over the leading dim
    ("hc.add_tensor", "tensor<2x2x2xi32>",
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


@pytest.mark.parametrize("op,ty,x,expected", [
    ("hc.relu", "i32", -3, 0),
    ("hc.relu_vec", "vector<4xi32>", [-1, 2, -3, 4], [0, 2, 0, 4]),
    ("hc.relu_tensor", "tensor<2x2xi32>", [[-1, 2], [3, -4]], [[0, 2], [3, 0]]),
])
def test_relu(op, ty, x, expected):
    check_oracle(f"""
    func.func @f(%x: {ty}) -> {ty} {{
      %r = "{op}"(%x) : ({ty}) -> {ty}
      func.return %r : {ty}
    }}""", [x], expected)


def test_mul_vec():
    """A scalar times every lane of a vector."""
    check_oracle("""
    func.func @f(%s: i32, %v: vector<4xi32>) -> vector<4xi32> {
      %r = "hc.mul_vec"(%s, %v) : (i32, vector<4xi32>) -> vector<4xi32>
      func.return %r : vector<4xi32>
    }""", [3, [1, 2, 3, 4]], [3, 6, 9, 12])


@pytest.mark.parametrize("a_shape,b_shape", [
    ((2, 3), (3, 4)),
    ((1, 5), (5, 1)),
    ((3, 1), (1, 2)),
    ((2, 2, 3), (2, 3, 4)),  # batched: one matmul per leading index
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
      %r = "hc.relu_tensor"(%m) : (tensor<2x4xi32>) -> tensor<2x4xi32>
      func.return %r : tensor<2x4xi32>
    }""", [a.tolist(), b.tolist()], np.maximum(a @ b, 0).tolist())


def test_read_before_write_in_place():
    """`%t` is read, then written. Bufferization stores into the same
    buffer, and the result is unchanged."""
    check_oracle("""
    func.func @f(%d: i32) -> i32 {
      %c0 = arith.constant 0 : index
      %t = arith.constant dense<5> : tensor<2x2xi32>
      %old = tensor.extract %t[%c0, %c0] : tensor<2x2xi32>
      %inc = arith.addi %old, %d : i32
      %t2 = tensor.insert %inc into %t[%c0, %c0] : tensor<2x2xi32>
      %x = tensor.extract %t2[%c0, %c0] : tensor<2x2xi32>
      func.return %x : i32
    }""", [3], 8)
