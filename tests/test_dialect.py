"""Construction + verify_() checks for hc dialect ops -- both valid
constructions and the type-mismatch cases each verify_() is meant to catch.
"""
import pytest

from xdsl.ir import Block
from conftest import vec_ty, const_i32, const_vec
from hc_dialect import (
    HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,
    HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec,
)

SCALAR_BINOPS = [HCAdd, HCMul, HCSub]
VECVEC_BINOPS = [HCAddVec, HCSubVec, HCMulVecVec]


@pytest.mark.parametrize("op_cls", SCALAR_BINOPS)
def test_scalar_binop_constructs(op_cls):
    a, b = const_i32(3), const_i32(4)
    op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
    op.verify()


def test_relu_constructs():
    x = const_i32(5)
    op = HCRelu(operands=[x.result], result_types=[x.result.type])
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
    with pytest.raises(Exception):
        op.verify()


def test_mul_vec_scalar_times_vector_constructs():
    s, v = const_i32(3), const_vec([1, 2, 3, 4])
    op = HCMulVec(operands=[s.result, v.result], result_types=[v.result.type])
    op.verify()


def test_mul_vec_rejects_result_type_mismatch():
    s, v = const_i32(3), const_vec([1, 2, 3, 4])
    op = HCMulVec(operands=[s.result, v.result], result_types=[vec_ty(8)])
    with pytest.raises(Exception):
        op.verify()


def test_relu_vec_constructs():
    v = const_vec([-1, 2, -3, 4])
    op = HCReluVec(operands=[v.result], result_types=[v.result.type])
    op.verify()


def test_relu_vec_rejects_result_type_mismatch():
    v = const_vec([-1, 2, -3, 4])
    op = HCReluVec(operands=[v.result], result_types=[vec_ty(8)])
    with pytest.raises(Exception):
        op.verify()


@pytest.mark.parametrize("op_cls", [HCPow, HCMax, HCMin])
def test_same_type_scalar_binop_constructs(op_cls):
    a, b = const_i32(2), const_i32(3)
    op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
    op.verify()
