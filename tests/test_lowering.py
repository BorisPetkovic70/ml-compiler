"""Structural checks: each hc.* op lowers to the expected arith/scf/vector
ops, and no hc.* op survives lowering.
"""
import pytest

from xdsl.dialects.builtin import i32
from xdsl.dialects import arith
from conftest import (
    build_module, const_i32, const_vec, vec_ty, tensor_ty, lower, entry_op_names,
    find_op, find_ops,
)
from hc_dialect import (
    HCAdd, HCMul, HCSub, HCRelu, HCMax, HCMin,
    HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec, HCMatmul,
    HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor,
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
    assert "arith.muli" in names
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

    cmp_op = find_op(m, "arith.cmpi")
    # Compare against xDSL's own canonical predicate encoding (built fresh
    # from the expected string) rather than hardcoding a duplicate int table.
    reference = arith.CmpiOp(cmp_op.operands[0], cmp_op.operands[1], expected_control_op)
    assert cmp_op.predicate == reference.predicate


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


# M, K, N all distinct so a swapped dimension anywhere in the nest can't hide.
_M, _K, _N = 2, 3, 4


def _matmul_module():
    """hc.matmul on two tensor block args; returns (module, A, B)."""
    captured = {}

    def body(args):
        a, b = args
        captured["a"], captured["b"] = a, b
        op = HCMatmul(operands=[a, b], result_types=[tensor_ty(_M, _N)])
        return [op], op.results[0]

    m = build_module([tensor_ty(_M, _K), tensor_ty(_K, _N)], body)
    return m, captured["a"], captured["b"]


def _index_const_value(v) -> int:
    return v.owner.value.value.data


def test_matmul_lowers_to_three_deep_scf_for_nest():
    m, _, _ = _matmul_module()
    lower(m)
    names = entry_op_names(m)
    assert not any(n.startswith("hc.") for n in names)
    assert names.count("scf.for") == 1  # one top-level loop; the rest are nested

    i_loop, j_loop, k_loop = find_ops(m, "scf.for")
    assert len(find_ops(m, "scf.for")) == 3
    assert k_loop.parent_op() is j_loop
    assert j_loop.parent_op() is i_loop
    assert i_loop.parent_op().name == "func.func"

    assert len(find_ops(m, "tensor.extract")) == 2
    assert len(find_ops(m, "tensor.insert")) == 1
    assert len(find_ops(m, "arith.muli")) == 1
    assert len(find_ops(m, "arith.addi")) == 1


def test_matmul_nest_bounds_and_wiring():
    """Structural presence can't prove the index wiring; assert on the actual
    operands. The semantic before/after check lands with the interpreter (M3)."""
    m, a, b = _matmul_module()
    lower(m)
    i_loop, j_loop, k_loop = find_ops(m, "scf.for")
    i, c_i = i_loop.body.block.args
    j, c_ij = j_loop.body.block.args
    k, acc = k_loop.body.block.args

    # trip counts: i over M, j over N, k over K
    assert [_index_const_value(l.ub) for l in (i_loop, j_loop, k_loop)] == [_M, _N, _K]
    assert all(_index_const_value(l.lb) == 0 and _index_const_value(l.step) == 1
               for l in (i_loop, j_loop, k_loop))

    # i/j thread the MxN tensor, k threads a scalar accumulator
    assert c_i.type == c_ij.type == tensor_ty(_M, _N)
    assert acc.type == i32

    # result tensor starts as dense zeros
    init = i_loop.iter_args[0].owner
    assert init.name == "arith.constant"
    assert init.result.type == tensor_ty(_M, _N)
    assert list(init.value.get_values()) == [0] * (_M * _N)
    # scalar accumulator starts at 0
    assert k_loop.iter_args[0].owner.value.value.data == 0

    # A[i,k] * B[k,j], accumulated into acc
    a_ex, b_ex = find_ops(m, "tensor.extract")
    assert a_ex.tensor is a and list(a_ex.indices) == [i, k]
    assert b_ex.tensor is b and list(b_ex.indices) == [k, j]
    (mul,) = find_ops(m, "arith.muli")
    assert list(mul.operands) == [a_ex.result, b_ex.result]
    (add,) = find_ops(m, "arith.addi")
    assert list(add.operands) == [acc, mul.result]

    # C[i,j] = k-loop result, into the j loop's tensor iter_arg
    (ins,) = find_ops(m, "tensor.insert")
    assert ins.scalar is k_loop.results[0]
    assert ins.dest is c_ij
    assert list(ins.indices) == [i, j]

    # loops chain the tensor outward, and the function returns the outer loop
    assert j_loop.iter_args[0] is c_i
    (ret,) = find_ops(m, "func.return")
    assert ret.operands[0] is i_loop.results[0]


def test_matmul_lowering_survives_fold_and_dce():
    """The default pipeline (lowering + folding + DCE) must leave the nest intact."""
    m, _, _ = _matmul_module()
    lower(m, fold=True, dce=True)
    assert len(find_ops(m, "scf.for")) == 3
    assert len(find_ops(m, "tensor.extract")) == 2
    assert len(find_ops(m, "tensor.insert")) == 1
    assert not any(n.startswith("hc.") for n in entry_op_names(m))


# ---------------- rank-2 tensor elementwise ops: 2-deep scf.for nest ----------------

# M, N distinct so a swapped dimension anywhere in the nest can't hide.
_TM, _TN = 2, 3


def _tensor_binop_module(op_cls):
    """op_cls(A, B) -> C, all tensor<_TMx_TNxi32>; returns (module, A, B)."""
    captured = {}

    def body(args):
        a, b = args
        captured["a"], captured["b"] = a, b
        op = op_cls(operands=[a, b], result_types=[tensor_ty(_TM, _TN)])
        return [op], op.results[0]

    m = build_module([tensor_ty(_TM, _TN), tensor_ty(_TM, _TN)], body)
    return m, captured["a"], captured["b"]


def _tensor_relu_module():
    captured = {}

    def body(args):
        (x,) = args
        captured["x"] = x
        op = HCReluTensor(operands=[x], result_types=[tensor_ty(_TM, _TN)])
        return [op], op.results[0]

    m = build_module([tensor_ty(_TM, _TN)], body)
    return m, captured["x"]


@pytest.mark.parametrize("op_cls,expected", [
    (HCAddTensor, "arith.addi"), (HCSubTensor, "arith.subi"), (HCMulTensor, "arith.muli"),
])
def test_tensor_binop_lowers_to_two_deep_scf_for_nest(op_cls, expected):
    m, _, _ = _tensor_binop_module(op_cls)
    lower(m)
    names = entry_op_names(m)
    assert not any(n.startswith("hc.") for n in names)
    assert names.count("scf.for") == 1  # one top-level loop; the other is nested

    i_loop, j_loop = find_ops(m, "scf.for")
    assert j_loop.parent_op() is i_loop
    assert i_loop.parent_op().name == "func.func"

    assert len(find_ops(m, "tensor.extract")) == 2
    assert len(find_ops(m, "tensor.insert")) == 1
    assert len(find_ops(m, expected)) == 1


def test_relu_tensor_lowers_to_two_deep_scf_for_nest_with_maxsi():
    m, _ = _tensor_relu_module()
    lower(m)
    names = entry_op_names(m)
    assert not any(n.startswith("hc.") for n in names)
    assert names.count("scf.for") == 1

    i_loop, j_loop = find_ops(m, "scf.for")
    assert j_loop.parent_op() is i_loop
    assert i_loop.parent_op().name == "func.func"

    assert len(find_ops(m, "tensor.extract")) == 1
    assert len(find_ops(m, "tensor.insert")) == 1
    assert len(find_ops(m, "arith.maxsi")) == 1


def test_tensor_binop_nest_wiring():
    """Structural presence can't prove the index wiring; assert on the actual
    operands, same discipline as the matmul nest's own wiring test."""
    m, a, b = _tensor_binop_module(HCAddTensor)
    lower(m)
    i_loop, j_loop = find_ops(m, "scf.for")
    i, c_i = i_loop.body.block.args
    j, c_ij = j_loop.body.block.args

    assert [_index_const_value(l.ub) for l in (i_loop, j_loop)] == [_TM, _TN]
    assert all(_index_const_value(l.lb) == 0 and _index_const_value(l.step) == 1
               for l in (i_loop, j_loop))
    assert c_i.type == c_ij.type == tensor_ty(_TM, _TN)

    a_ex, b_ex = find_ops(m, "tensor.extract")
    assert a_ex.tensor is a and list(a_ex.indices) == [i, j]
    assert b_ex.tensor is b and list(b_ex.indices) == [i, j]

    (ins,) = find_ops(m, "tensor.insert")
    assert ins.dest is c_ij
    assert list(ins.indices) == [i, j]

    assert j_loop.iter_args[0] is c_i
    (ret,) = find_ops(m, "func.return")
    assert ret.operands[0] is i_loop.results[0]


@pytest.mark.parametrize("op_cls", [HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor])
def test_tensor_binop_lowering_survives_fold_and_dce(op_cls):
    if op_cls is HCReluTensor:
        m, _ = _tensor_relu_module()
    else:
        m, _, _ = _tensor_binop_module(op_cls)
    lower(m, fold=True, dce=True)
    assert len(find_ops(m, "scf.for")) == 2
    assert len(find_ops(m, "tensor.insert")) == 1
    assert not any(n.startswith("hc.") for n in entry_op_names(m))


def test_chained_matmul_then_relu_tensor_lowering():
    """hc.matmul feeding hc.relu_tensor: proves the two nests actually chain
    (the relu nest reads the matmul nest's result), not just coexist."""
    def body(args):
        a, b = args
        mm = HCMatmul(operands=[a, b], result_types=[tensor_ty(_M, _N)])
        relu = HCReluTensor(operands=[mm.results[0]], result_types=[tensor_ty(_M, _N)])
        return [mm, relu], relu.results[0]

    m = build_module([tensor_ty(_M, _K), tensor_ty(_K, _N)], body)
    lower(m)
    names = entry_op_names(m)
    assert not any(n.startswith("hc.") for n in names)
    assert len(find_ops(m, "scf.for")) == 5  # 3 (matmul) + 2 (relu_tensor)

    # Both nests' outer loops are top-level (sequential) ops in the function body;
    # matmul's is emitted first, relu_tensor's second.
    top_loops = [l for l in find_ops(m, "scf.for") if l.parent_op().name == "func.func"]
    assert len(top_loops) == 2
    matmul_top_loop, relu_top_loop = top_loops

    (ret,) = find_ops(m, "func.return")
    relu_extract = find_ops(m, "tensor.extract")[-1]  # relu's own extract, last emitted
    assert relu_extract.tensor is matmul_top_loop.results[0]
    assert ret.operands[0] is relu_top_loop.results[0]
    assert ret.operands[0] is not matmul_top_loop.results[0]  # returns relu's result, not matmul's
