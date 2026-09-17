"""Structural checks: each hc.* op lowers to the expected arith/scf/vector
ops, and no hc.* op survives lowering.
"""
import pytest

from xdsl.dialects.builtin import i32
from conftest import build_module, const_i32, const_vec, vec_ty, lower, entry_op_names
from hc_dialect import (
    HCAdd, HCMul, HCSub, HCRelu, HCMax, HCMin,
    HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec,
)


def _scalar_binop(op_cls, expected_op):
    def body(_args):
        a, b = const_i32(3), const_i32(4)
        op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
        return [a, b, op], op.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert expected_op in names
    assert not any(n.startswith("hc.") for n in names)


@pytest.mark.parametrize("op_cls,expected", [
    (HCAdd, "arith.addi"), (HCMul, "arith.muli"), (HCSub, "arith.subi"),
])
def test_scalar_binop_lowering(op_cls, expected):
    _scalar_binop(op_cls, expected)


def test_relu_lowers_to_maxsi_with_zero():
    def body(_args):
        x = const_i32(5)
        r = HCRelu(operands=[x.result], result_types=[x.result.type])
        return [x, r], r.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert "arith.maxsi" in names
    assert not any(n.startswith("hc.") for n in names)


@pytest.mark.parametrize("op_cls,expected", [
    (HCAddVec, "arith.addi"), (HCSubVec, "arith.subi"), (HCMulVecVec, "arith.muli"),
])
def test_vecvec_binop_lowering(op_cls, expected):
    def body(_args):
        a, b = const_vec([1, 2, 3, 4]), const_vec([5, 6, 7, 8])
        op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
        return [a, b, op], op.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert expected in names
    assert not any(n.startswith("hc.") for n in names)


def test_mul_vec_constant_scalar_lowers_via_broadcast():
    def body(_args):
        s, v = const_i32(3), const_vec([1, 2, 3, 4])
        op = HCMulVec(operands=[s.result, v.result], result_types=[v.result.type])
        return [s, v, op], op.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert "vector.broadcast" in names
    assert "arith.muli" in names
    assert not any(n.startswith("hc.") for n in names)


def test_mul_vec_runtime_scalar_lowers_via_broadcast():
    """The case that motivated switching from constant-splat to vector.broadcast:
    the scalar is a function argument, not an arith.constant."""
    def body(args):
        (s,) = args
        v = const_vec([1, 2, 3, 4])
        op = HCMulVec(operands=[s, v.result], result_types=[v.result.type])
        return [v, op], op.results[0]
    m = build_module([i32], body)
    lower(m)
    names = entry_op_names(m)
    assert "vector.broadcast" in names
    assert not any(n.startswith("hc.") for n in names)


def test_relu_vec_lowers_to_maxsi_with_dense_zero():
    def body(_args):
        v = const_vec([-1, 2, -3, 4])
        r = HCReluVec(operands=[v.result], result_types=[v.result.type])
        return [v, r], r.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert "arith.maxsi" in names
    assert not any(n.startswith("hc.") for n in names)


@pytest.mark.parametrize("op_cls,expected_control_op", [(HCMax, "sgt"), (HCMin, "slt")])
def test_max_min_lower_to_cmpi_scf_if(op_cls, expected_control_op):
    def body(_args):
        a, b = const_i32(3), const_i32(4)
        op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
        return [a, b, op], op.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert "arith.cmpi" in names
    assert "scf.if" in names
    assert not any(n.startswith("hc.") for n in names)


def test_pow_lowers_to_scf_for_with_index_cast():
    def body(_args):
        base, exp = const_i32(2), const_i32(3)
        from hc_dialect import HCPow
        op = HCPow(operands=[base.result, exp.result], result_types=[base.result.type])
        return [base, exp, op], op.results[0]
    m = build_module([], body)
    lower(m)
    names = entry_op_names(m)
    assert "scf.for" in names
    assert "arith.index_cast" in names
    assert not any(n.startswith("hc.") for n in names)
