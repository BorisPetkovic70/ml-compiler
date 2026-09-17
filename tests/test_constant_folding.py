"""Constant folding: scalar and vector arith folds to a single constant, and
vector.broadcast of a constant scalar folds into a dense vector constant."""
from xdsl.dialects.builtin import i32
from xdsl.dialects import arith, vector

from conftest import build_module, const_i32, const_vec, vec_ty, lower, entry_op_names, run


def test_scalar_chain_folds_to_single_constant():
    """(3 + 4) * 2 -> a single arith.constant 14."""
    def body(_args):
        c1, c2 = const_i32(3), const_i32(4)
        add = arith.AddiOp(c1.result, c2.result)
        c3 = const_i32(2)
        mul = arith.MuliOp(add.result, c3.result)
        return [c1, c2, add, c3, mul], mul.result

    m = build_module([], body)
    lower(m, fold=True)
    assert entry_op_names(m) == ["arith.constant", "func.return"]
    assert run(m) == 14


def test_vector_chain_folds_to_single_constant():
    """relu(([1,2,3,4]+[10,20,30,40]) * [-1,-1,-1,-1]) -> dense<0>, since the
    product is all-negative."""
    def body(_args):
        v1, v2 = const_vec([1, 2, 3, 4]), const_vec([10, 20, 30, 40])
        add = arith.AddiOp(v1.result, v2.result)
        v3 = const_vec([-1, -1, -1, -1])
        mul = arith.MuliOp(add.result, v3.result)
        v0 = const_vec([0, 0, 0, 0])
        mx = arith.MaxSIOp(mul.result, v0.result)
        return [v1, v2, add, v3, mul, v0, mx], mx.result

    m = build_module([], body)
    lower(m, fold=True)
    assert entry_op_names(m) == ["arith.constant", "func.return"]
    assert run(m) == [0, 0, 0, 0]


def test_broadcast_of_constant_scalar_folds_to_dense_vector():
    """vector.broadcast(const 3) then muli with [1,2,3,4] folds all the way
    to dense<[3,6,9,12]>, with DCE cleaning up every intermediate op."""
    def body(_args):
        c = const_i32(3)
        v = const_vec([1, 2, 3, 4])
        bcast = vector.BroadcastOp(c.result, vec_ty(4))
        mul = arith.MuliOp(v.result, bcast.vector)
        return [c, v, bcast, mul], mul.result

    m = build_module([], body)
    lower(m, fold=True, dce=True)
    assert entry_op_names(m) == ["arith.constant", "func.return"]
    assert run(m) == [3, 6, 9, 12]


def test_no_fold_when_operand_is_runtime_value():
    """A runtime block-arg operand must NOT be folded away."""
    def body(args):
        (s,) = args
        c = const_i32(4)
        add = arith.AddiOp(s, c.result)
        return [c, add], add.result

    m = build_module([i32], body)
    lower(m, fold=True)
    assert "arith.addi" in entry_op_names(m)
    assert run(m, args=[10]) == 14
