"""Dead code elimination: unused arith ops (and vector.broadcast) are
removed; used ops and func.return survive."""
from xdsl.dialects.builtin import i32
from xdsl.dialects import arith, vector

from conftest import build_module, const_i32, vec_ty, lower, entry_op_names, run


def test_unused_arith_op_is_removed():
    def body(_args):
        used = const_i32(5)
        unused = const_i32(999)  # never consumed
        return [used, unused], used.result

    m = build_module([], body)
    lower(m, dce=True)
    names = entry_op_names(m)
    assert names.count("arith.constant") == 1
    assert run(m) == 5


def test_used_op_survives_dce():
    def body(_args):
        c1, c2 = const_i32(3), const_i32(4)
        add = arith.AddiOp(c1.result, c2.result)
        return [c1, c2, add], add.result

    m = build_module([], body)
    lower(m, dce=True)
    assert "arith.addi" in entry_op_names(m)
    assert run(m) == 7


def test_unused_vector_broadcast_is_removed():
    """A broadcast whose source is a runtime value (survives folding) but
    whose result is never used should still be dropped by DCE."""
    def body(args):
        (s,) = args
        bcast = vector.BroadcastOp(s, vec_ty(4))
        return [bcast], s

    m = build_module([i32], body)
    lower(m, dce=True)
    assert "vector.broadcast" not in entry_op_names(m)
    assert run(m, args=[7]) == 7
