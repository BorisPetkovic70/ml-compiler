"""Operator semantics: run each op through the interpreter both BEFORE and
AFTER lowering and assert the same result -- this is what actually proves a
lowering is correct, without compiling anything.
"""
import pytest

from xdsl.dialects.builtin import i32
from conftest import build_module, const_i32, const_vec, run, lower
from hc_dialect import (
    HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,
    HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec,
)


def _scalar_binop_module(op_cls, a_val, b_val):
    def body(_args):
        a, b = const_i32(a_val), const_i32(b_val)
        op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
        return [a, b, op], op.results[0]
    return build_module([], body)


@pytest.mark.parametrize("op_cls,a,b,expected", [
    (HCAdd, 3, 4, 7),
    (HCMul, 3, 4, 12),
    (HCSub, 10, 4, 6),
    (HCMax, 3, 9, 9),
    (HCMin, 3, 9, 3),
    (HCPow, 2, 5, 32),
])
def test_scalar_binop_semantics_before_and_after_lowering(op_cls, a, b, expected):
    m = _scalar_binop_module(op_cls, a, b)
    assert run(m) == expected
    lower(m)
    assert run(m) == expected


def test_relu_semantics_before_and_after_lowering():
    for val, expected in [(5, 5), (-3, 0), (0, 0)]:
        def body(_args, val=val):
            x = const_i32(val)
            r = HCRelu(operands=[x.result], result_types=[x.result.type])
            return [x, r], r.results[0]
        m = build_module([], body)
        assert run(m) == expected
        lower(m)
        assert run(m) == expected


def _vecvec_binop_module(op_cls, a_vals, b_vals):
    def body(_args):
        a, b = const_vec(a_vals), const_vec(b_vals)
        op = op_cls(operands=[a.result, b.result], result_types=[a.result.type])
        return [a, b, op], op.results[0]
    return build_module([], body)


@pytest.mark.parametrize("op_cls,a,b,expected", [
    (HCAddVec, [1, 2, 3, 4], [10, 20, 30, 40], [11, 22, 33, 44]),
    (HCSubVec, [10, 20, 30, 40], [1, 2, 3, 4], [9, 18, 27, 36]),
    (HCMulVecVec, [1, 2, 3, 4], [2, 3, 4, 5], [2, 6, 12, 20]),
])
def test_vecvec_binop_semantics_before_and_after_lowering(op_cls, a, b, expected):
    m = _vecvec_binop_module(op_cls, a, b)
    assert run(m) == expected
    lower(m)
    assert run(m) == expected


def test_relu_vec_semantics_before_and_after_lowering():
    def body(_args):
        v = const_vec([-1, 2, -3, 4])
        r = HCReluVec(operands=[v.result], result_types=[v.result.type])
        return [v, r], r.results[0]
    m = build_module([], body)
    expected = [0, 2, 0, 4]
    assert run(m) == expected
    lower(m)
    assert run(m) == expected


def test_mul_vec_constant_scalar_semantics_before_and_after_lowering():
    def body(_args):
        s, v = const_i32(3), const_vec([1, 2, 3, 4])
        op = HCMulVec(operands=[s.result, v.result], result_types=[v.result.type])
        return [s, v, op], op.results[0]
    m = build_module([], body)
    expected = [3, 6, 9, 12]
    assert run(m) == expected
    lower(m)
    assert run(m) == expected


def test_mul_vec_runtime_scalar_semantics_before_and_after_lowering():
    """The scalar is a function argument (not a constant) -- the case that
    required switching the lowering from constant-splat to vector.broadcast."""
    def body(args):
        (s,) = args
        v = const_vec([1, 2, 3, 4])
        op = HCMulVec(operands=[s, v.result], result_types=[v.result.type])
        return [v, op], op.results[0]
    m = build_module([i32], body)
    expected = [100, 200, 300, 400]
    assert run(m, args=[100]) == expected
    lower(m)
    assert run(m, args=[100]) == expected


def test_mixed_scalar_and_vector_args_affine_relu():
    """y = relu(a * x + b) with a scalar, x/b vector -- exercises multiple
    inputs, mixed scalar+vector, and the full add/mul/relu chain together."""
    def body(args):
        x, a, b = args
        mul = HCMulVec(operands=[a, x], result_types=[x.type])
        add = HCAddVec(operands=[mul.results[0], b], result_types=[x.type])
        relu = HCReluVec(operands=[add.results[0]], result_types=[x.type])
        return [mul, add, relu], relu.results[0]

    from conftest import vec_ty
    m = build_module([vec_ty(4), i32, vec_ty(4)], body)
    x, a, b = [1, 2, 3, 4], 20, [-30, 0, 5, 100]
    expected = [max(a * x[i] + b[i], 0) for i in range(4)]
    assert run(m, args=[x, a, b]) == expected
    lower(m)
    assert run(m, args=[x, a, b]) == expected
