"""Construction + verify_() checks for hc dialect ops -- both valid
constructions and the type-mismatch cases each verify_() is meant to catch.
"""
import pytest

from xdsl.ir import Block
from xdsl.dialects.builtin import IntegerType, TensorType
from conftest import vec_ty, tensor_ty, const_i32, const_vec
from hc_dialect import (
    HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,
    HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec,
    HCMatmul, HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor,
    broadcast_shape,
)

SCALAR_BINOPS = [HCAdd, HCMul, HCSub]
VECVEC_BINOPS = [HCAddVec, HCSubVec, HCMulVecVec]
TENSOR_BINOPS = [HCAddTensor, HCSubTensor, HCMulTensor]


@pytest.mark.parametrize("op_cls", SCALAR_BINOPS)
def test_scalar_binop_constructs(op_cls):
    a, b = const_i32(3), const_i32(4)
    op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
    op.verify()


@pytest.mark.parametrize("op_cls", SCALAR_BINOPS)
def test_scalar_binop_rejects_mismatched_operand_types(op_cls):
    """IRDL's bare operand_def(IntegerType) doesn't bind operand widths
    together, so a mismatched-width pair only verify_() catches."""
    blk = Block(arg_types=[IntegerType(32), IntegerType(16)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[IntegerType(32)])
    with pytest.raises(ValueError, match="same integer type"):
        op.verify()


@pytest.mark.parametrize("op_cls", SCALAR_BINOPS)
def test_scalar_binop_rejects_mismatched_result_type(op_cls):
    """Operands match each other but the result type differs."""
    a, b = const_i32(2), const_i32(3)
    op = op_cls(operands=[a.result, b.result], result_types=[IntegerType(16)])
    with pytest.raises(ValueError, match="same integer type"):
        op.verify()


def test_relu_constructs():
    x = const_i32(5)
    op = HCRelu(operands=[x.result], result_types=[x.result.type])
    op.verify()


def test_relu_rejects_result_type_mismatch():
    x = const_i32(5)
    op = HCRelu(operands=[x.result], result_types=[IntegerType(16)])
    with pytest.raises(ValueError, match="must match operand type"):
        op.verify()


@pytest.mark.parametrize("op_cls", VECVEC_BINOPS)
def test_vecvec_binop_constructs(op_cls):
    a, b = const_vec([1, 2, 3, 4]), const_vec([5, 6, 7, 8])
    op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
    op.verify()


@pytest.mark.parametrize("op_cls", VECVEC_BINOPS)
def test_vecvec_binop_rejects_mismatched_shapes(op_cls):
    blk = Block(arg_types=[vec_ty(4), vec_ty(8)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[vec_ty(4)])
    with pytest.raises(ValueError, match="same vector type"):
        op.verify()


def test_mul_vec_scalar_times_vector_constructs():
    s, v = const_i32(3), const_vec([1, 2, 3, 4])
    op = HCMulVec(operands=[s.result, v.result], result_types=[v.result.type])
    op.verify()


def test_mul_vec_rejects_result_type_mismatch():
    s, v = const_i32(3), const_vec([1, 2, 3, 4])
    op = HCMulVec(operands=[s.result, v.result], result_types=[vec_ty(8)])
    with pytest.raises(ValueError, match="match vector operand type"):
        op.verify()


def test_relu_vec_constructs():
    v = const_vec([-1, 2, -3, 4])
    op = HCReluVec(operands=[v.result], result_types=[v.result.type])
    op.verify()


def test_relu_vec_rejects_result_type_mismatch():
    v = const_vec([-1, 2, -3, 4])
    op = HCReluVec(operands=[v.result], result_types=[vec_ty(8)])
    with pytest.raises(ValueError, match="match operand vector type"):
        op.verify()


@pytest.mark.parametrize("op_cls", [HCPow, HCMax, HCMin])
def test_same_type_scalar_binop_constructs(op_cls):
    a, b = const_i32(2), const_i32(3)
    op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
    op.verify()


@pytest.mark.parametrize("op_cls", [HCPow, HCMax, HCMin])
def test_same_type_scalar_binop_rejects_mismatched_operand_types(op_cls):
    """IRDL's bare operand_def(IntegerType) doesn't bind operand widths
    together, so a mismatched-width pair only verify_() catches."""
    blk = Block(arg_types=[IntegerType(32), IntegerType(16)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[IntegerType(32)])
    with pytest.raises(ValueError, match="same integer type"):
        op.verify()


@pytest.mark.parametrize("op_cls", [HCPow, HCMax, HCMin])
def test_same_type_scalar_binop_rejects_mismatched_result_type(op_cls):
    """Operands match each other but the result type differs."""
    a, b = const_i32(2), const_i32(3)
    op = op_cls(operands=[a.result, b.result], result_types=[IntegerType(16)])
    with pytest.raises(ValueError, match="same integer type"):
        op.verify()


@pytest.mark.parametrize("op_cls", VECVEC_BINOPS)
def test_vecvec_binop_rejects_mismatched_result_type(op_cls):
    """lhs and rhs match each other, but the result type differs from both --
    the second raise branch in _verify_bin_same_vec_type."""
    a, b = const_vec([1, 2, 3, 4]), const_vec([5, 6, 7, 8])
    op = op_cls(operands=[a.result, b.result], result_types=[vec_ty(8)])
    with pytest.raises(ValueError, match="match operand vector type"):
        op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_constructs(op_cls):
    blk = Block(arg_types=[tensor_ty(2, 3), tensor_ty(2, 3)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(2, 3)])
    op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_constructs_rank_1(op_cls):
    blk = Block(arg_types=[tensor_ty(4), tensor_ty(4)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(4)])
    op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_rejects_mismatched_shapes(op_cls):
    blk = Block(arg_types=[tensor_ty(2, 3), tensor_ty(2, 4)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(2, 3)])
    with pytest.raises(ValueError, match="cannot broadcast shapes"):
        op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_rejects_mismatched_result_type(op_cls):
    """The operands broadcast to 2x3, but the result is declared 2x4."""
    blk = Block(arg_types=[tensor_ty(2, 3), tensor_ty(2, 3)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(2, 4)])
    with pytest.raises(ValueError, match="result shape must be"):
        op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_constructs_batched(op_cls):
    blk = Block(arg_types=[tensor_ty(2, 3, 4), tensor_ty(2, 3, 4)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(2, 3, 4)])
    op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_rejects_mismatched_batch_dim(op_cls):
    blk = Block(arg_types=[tensor_ty(2, 3, 4), tensor_ty(5, 3, 4)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(2, 3, 4)])
    with pytest.raises(ValueError, match="cannot broadcast shapes"):
        op.verify()


@pytest.mark.parametrize("a,b,expected", [
    ([2, 3], [2, 3], [2, 3]),
    ([2, 3], [3], [2, 3]),        # aligned at the last dim
    ([2, 1], [1, 3], [2, 3]),     # each operand stretches one dim
    ([], [4], [4]),               # a scalar has shape []
])
def test_broadcast_shape(a, b, expected):
    assert broadcast_shape(a, b) == expected


def test_broadcast_shape_rejects_unequal_dims():
    """Dims of 3 and 4 differ and neither is 1."""
    with pytest.raises(ValueError, match="cannot broadcast shapes"):
        broadcast_shape([2, 3], [4])


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_constructs_with_broadcast(op_cls):
    blk = Block(arg_types=[tensor_ty(2, 3), tensor_ty(3)])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty(2, 3)])
    op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_constructs_with_scalar_operand(op_cls):
    blk = Block(arg_types=[IntegerType(32), tensor_ty(2, 3)])
    s, t = blk.args
    op = op_cls(operands=[s, t], result_types=[tensor_ty(2, 3)])
    op.verify()


@pytest.mark.parametrize("op_cls", TENSOR_BINOPS)
def test_tensor_binop_rejects_rank_0_operand(op_cls):
    """`tensor<i32>` has no dim to loop over; a single value is an `i32`."""
    blk = Block(arg_types=[tensor_ty(), tensor_ty()])
    a, b = blk.args
    op = op_cls(operands=[a, b], result_types=[tensor_ty()])
    with pytest.raises(ValueError, match="rank >= 1"):
        op.verify()


def test_relu_tensor_constructs():
    blk = Block(arg_types=[tensor_ty(2, 3)])
    (x,) = blk.args
    op = HCReluTensor(operands=[x], result_types=[tensor_ty(2, 3)])
    op.verify()


def test_relu_tensor_constructs_rank_1():
    blk = Block(arg_types=[tensor_ty(4)])
    (x,) = blk.args
    op = HCReluTensor(operands=[x], result_types=[tensor_ty(4)])
    op.verify()


def test_relu_tensor_rejects_result_type_mismatch():
    blk = Block(arg_types=[tensor_ty(2, 3)])
    (x,) = blk.args
    op = HCReluTensor(operands=[x], result_types=[tensor_ty(2, 4)])
    with pytest.raises(ValueError, match="match operand tensor type"):
        op.verify()


def test_relu_tensor_constructs_batched():
    blk = Block(arg_types=[tensor_ty(2, 3, 4)])
    (x,) = blk.args
    op = HCReluTensor(operands=[x], result_types=[tensor_ty(2, 3, 4)])
    op.verify()


def test_relu_tensor_rejects_rank_0_operand():
    blk = Block(arg_types=[tensor_ty()])
    (x,) = blk.args
    op = HCReluTensor(operands=[x], result_types=[tensor_ty()])
    with pytest.raises(ValueError, match="rank >= 1"):
        op.verify()


def test_matmul_constructs():
    blk = Block(arg_types=[tensor_ty(4, 6), tensor_ty(6, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(4, 8)])
    op.verify()


def test_matmul_rejects_non_rank2_operand():
    blk = Block(arg_types=[TensorType(IntegerType(32), [4]), tensor_ty(6, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(4, 8)])
    with pytest.raises(ValueError, match="rank-2, or rank-3"):
        op.verify()


def test_matmul_rejects_mismatched_inner_dim():
    blk = Block(arg_types=[tensor_ty(4, 6), tensor_ty(7, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(4, 8)])
    with pytest.raises(ValueError, match="inner dimensions must agree"):
        op.verify()


def test_matmul_rejects_mismatched_result_shape():
    blk = Block(arg_types=[tensor_ty(4, 6), tensor_ty(6, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(4, 9)])
    with pytest.raises(ValueError, match="result shape must be"):
        op.verify()


def test_matmul_rejects_mismatched_element_type():
    blk = Block(arg_types=[
        TensorType(IntegerType(32), [4, 6]),
        TensorType(IntegerType(16), [6, 8]),
    ])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(4, 8)])
    with pytest.raises(ValueError, match="same element type"):
        op.verify()


def test_matmul_constructs_batched():
    blk = Block(arg_types=[tensor_ty(2, 4, 6), tensor_ty(2, 6, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(2, 4, 8)])
    op.verify()


def test_matmul_rejects_mismatched_batch_dim():
    blk = Block(arg_types=[tensor_ty(2, 4, 6), tensor_ty(3, 6, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(2, 4, 8)])
    with pytest.raises(ValueError, match="batch dimensions must match"):
        op.verify()


def test_matmul_rejects_mismatched_ranks():
    """lhs is rank-2, rhs is rank-3 -- ambiguous whether the whole op is
    batched, so this is rejected rather than guessed at."""
    blk = Block(arg_types=[tensor_ty(4, 6), tensor_ty(2, 6, 8)])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(2, 4, 8)])
    with pytest.raises(ValueError, match="must all be the same rank"):
        op.verify()


def test_matmul_rejects_rank_4_operand():
    blk = Block(arg_types=[
        TensorType(IntegerType(32), [1, 2, 4, 6]), tensor_ty(2, 6, 8),
    ])
    a, b = blk.args
    op = HCMatmul(operands=[a, b], result_types=[tensor_ty(2, 4, 8)])
    with pytest.raises(ValueError, match="rank-2, or rank-3"):
        op.verify()
