"""Structural checks for tensor -> memref bufferization.

Most tests call apply_bufferization directly on an already-lowered module, to
isolate the pass from the rest of the pipeline. They check both which ops are
present and how their operands are wired.
test_pipeline_bufferize_flag_wires_in_the_pass covers the pipeline wiring. The
semantic (NumPy-equality) check of bufferized IR is in test_interpreter.py.
"""
import pytest

from xdsl.dialects import arith, func, scf, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr, IndexType, MemRefType, ModuleOp, i32,
)
from xdsl.ir import Block, Region

from conftest import build_module, tensor_ty, lower, find_ops
from hc_dialect import HCMatmul, HCReluTensor
from middle_end.bufferization import apply_bufferization

# M, K, N all distinct so a swapped dimension anywhere in the nest can't hide.
_M, _K, _N = 2, 3, 4


def _matmul_module():
    """hc.matmul on two tensor block args, lowered to its scf.for nest; returns (module, A, B)."""
    captured = {}

    def body(args):
        a, b = args
        captured["a"], captured["b"] = a, b
        op = HCMatmul(operands=[a, b], result_types=[tensor_ty(_M, _N)])
        return [op], op.results[0]

    m = build_module([tensor_ty(_M, _K), tensor_ty(_K, _N)], body)
    lower(m)
    return m, captured["a"], captured["b"]


def test_bufferized_signature_is_all_memref():
    m, _, _ = _matmul_module()
    apply_bufferization(m)
    m.verify()
    fn = next(op for b in m.body.blocks for op in b.ops if isinstance(op, func.FuncOp))
    assert all(isinstance(t, MemRefType) for t in fn.function_type.inputs)
    assert all(isinstance(t, MemRefType) for t in fn.function_type.outputs)
    assert all(isinstance(a.type, MemRefType) for a in fn.body.block.args)


def test_no_tensor_ops_survive_bufferization():
    m, _, _ = _matmul_module()
    apply_bufferization(m)
    m.verify()
    names = [op.name for op in m.walk()]
    assert not any(n.startswith("tensor.") for n in names)
    assert names.count("memref.alloc") == 1
    assert names.count("memref.load") == 2     # one per tensor.extract in the lowered nest
    assert names.count("memref.store") == 2    # 1 for the zero-fill, 1 for the real store


def _batched_matmul_module():
    """hc.matmul on two rank-3 (batched) tensor block args, lowered; returns (module, A, B)."""
    captured = {}

    def body(args):
        a, b = args
        captured["a"], captured["b"] = a, b
        op = HCMatmul(operands=[a, b], result_types=[tensor_ty(2, _M, _N)])
        return [op], op.results[0]

    m = build_module([tensor_ty(2, _M, _K), tensor_ty(2, _K, _N)], body)
    lower(m)
    return m, captured["a"], captured["b"]


def test_batched_tensor_bufferizes_with_the_same_op_counts_as_rank_2():
    """A rank-3 (batched) matmul bufferizes to the same alloc/load/store
    counts as rank 2, with 3-index loads and memref-only signature types."""
    m, _, _ = _batched_matmul_module()
    apply_bufferization(m)
    m.verify()
    names = [op.name for op in m.walk()]
    assert not any(n.startswith("tensor.") for n in names)
    assert names.count("memref.alloc") == 1
    assert names.count("memref.load") == 2
    assert names.count("memref.store") == 2  # 1 zero-fill, 1 real store

    fn = next(op for blk in m.body.blocks for op in blk.ops if isinstance(op, func.FuncOp))
    assert all(isinstance(t, MemRefType) for t in fn.function_type.inputs)
    assert all(isinstance(t, MemRefType) for t in fn.function_type.outputs)

    a_ld, b_ld = find_ops(m, "memref.load")
    assert len(a_ld.indices) == 3  # batch, i, k
    assert len(b_ld.indices) == 3  # batch, k, j


def test_i_and_j_loops_drop_iter_args_k_loop_keeps_its_accumulator():
    """After bufferization, stores mutate one buffer, so the i and j loops carry
    no iter_args. The k loop keeps its scalar accumulator."""
    m, _, _ = _matmul_module()
    apply_bufferization(m)
    m.verify()
    # Two scf.for nests remain: the zero-fill (2 loops) and the matmul nest (3 loops).
    fill_i, fill_j, i_loop, j_loop, k_loop = find_ops(m, "scf.for")

    assert len(fill_i.iter_args) == 0 and len(fill_j.iter_args) == 0
    assert len(i_loop.iter_args) == 0 and len(i_loop.results) == 0
    assert len(j_loop.iter_args) == 0 and len(j_loop.results) == 0
    assert len(k_loop.iter_args) == 1 and k_loop.iter_args[0].type == i32
    assert i_loop.body.block.args[0].type == j_loop.body.block.args[0].type == IndexType()


def test_load_store_indices_and_store_value_match_the_lowering():
    """Structural presence can't prove the wiring is correct: a load/store could sit
    in the right place with the wrong indices. Assert on the actual operands instead."""
    m, a, b = _matmul_module()
    apply_bufferization(m)
    m.verify()
    fn = next(op for blk in m.body.blocks for op in blk.ops if isinstance(op, func.FuncOp))
    a_buf, b_buf = fn.body.block.args
    *_, i_loop, j_loop, k_loop = find_ops(m, "scf.for")
    i = i_loop.body.block.args[0]
    j = j_loop.body.block.args[0]
    k = k_loop.body.block.args[0]

    a_ld, b_ld = find_ops(m, "memref.load")
    assert a_ld.memref is a_buf and list(a_ld.indices) == [i, k]
    assert b_ld.memref is b_buf and list(b_ld.indices) == [k, j]

    (mul,) = find_ops(m, "arith.muli")
    assert list(mul.operands) == [a_ld.results[0], b_ld.results[0]]

    result_store = find_ops(m, "memref.store")[1]     # [0] is the zero-fill's store
    assert list(result_store.indices) == [i, j]
    assert result_store.value is k_loop.results[0]

    (ret,) = find_ops(m, "func.return")
    assert ret.operands[0] is result_store.memref


def test_zero_fill_store_indices_match_the_loop_order():
    """The fill nest's outer loop must range over dim 0 (rows) and its store must
    index [row_iv, col_iv] in that order -- swapped, a non-square buffer like this
    2x4 one would be walked out of its declared shape."""
    m, _, _ = _matmul_module()
    apply_bufferization(m)
    m.verify()
    fill_i, fill_j, *_ = find_ops(m, "scf.for")
    assert _index_const_value(fill_i.ub) == _M
    assert _index_const_value(fill_j.ub) == _N
    (fill_store,) = [s for s in find_ops(m, "memref.store") if s.parent_op() is fill_j]
    assert list(fill_store.indices) == [fill_i.body.block.args[0], fill_j.body.block.args[0]]


def test_result_buffer_is_returned_not_deallocated():
    m, _, _ = _matmul_module()
    apply_bufferization(m)
    m.verify()
    assert len(find_ops(m, "memref.dealloc")) == 0
    (alloc,) = find_ops(m, "memref.alloc")
    (ret,) = find_ops(m, "func.return")
    assert ret.operands[0] is alloc.memref


def test_function_without_tensors_is_left_untouched():
    def body(args):
        a, b = args
        op = arith.AddiOp(a, b)
        return [op], op.results[0]
    m = build_module([i32, i32], body)
    before = str(m)
    apply_bufferization(m)
    assert str(m) == before


def test_chained_matmul_then_relu_tensor_intermediate_buffer_is_threaded_and_freed():
    """hc.matmul's result is a temporary buffer read only by hc.relu_tensor.
    The relu nest must memref.load from that buffer, and the buffer must be
    deallocated, not leaked or aliased into the returned buffer."""
    def body(args):
        a, b = args
        mm = HCMatmul(operands=[a, b], result_types=[tensor_ty(_M, _N)])
        relu = HCReluTensor(operands=[mm.results[0]], result_types=[tensor_ty(_M, _N)])
        return [mm, relu], relu.results[0]

    m = build_module([tensor_ty(_M, _K), tensor_ty(_K, _N)], body)
    lower(m)
    apply_bufferization(m)
    m.verify()

    allocs = find_ops(m, "memref.alloc")
    assert len(allocs) == 2  # one buffer per tensor value: matmul's result, relu's result

    (ret,) = find_ops(m, "func.return")
    returned_buf = ret.operands[0]
    intermediate_buf = next(a.memref for a in allocs if a.memref is not returned_buf)

    # relu's nest reads the matmul buffer directly, not some other value.
    relu_loads = [ld for ld in find_ops(m, "memref.load") if ld.memref is intermediate_buf]
    assert len(relu_loads) == 1

    (dealloc,) = find_ops(m, "memref.dealloc")
    assert dealloc.memref is intermediate_buf
    assert returned_buf is not intermediate_buf


# ---- refusal guards: each must fail loudly rather than silently miscompile --------

def _idx(v: int):
    return arith.ConstantOp.from_int_and_width(v, IndexType())


def _index_const_value(v) -> int:
    return v.owner.value.value.data


def test_refuses_insert_into_a_tensor_with_a_second_use():
    """Two inserts share one source tensor, so storing into it in place for one
    would corrupt what the other sees."""
    def body(_args):
        t0 = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 2), [0] * 4))
        i0, i1 = _idx(0), _idx(1)
        v = arith.ConstantOp.from_int_and_width(9, 32)
        t1 = tensor.InsertOp(v.result, t0.result, [i0.result, i0.result])
        t2 = tensor.InsertOp(v.result, t0.result, [i1.result, i1.result])  # 2nd use of t0
        return [t0, i0, i1, v, t1, t2], t2.results[0]
    m = build_module([], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="used again elsewhere"):
        apply_bufferization(m)


def test_refuses_insert_into_a_function_argument():
    """Bufferizing this in place would mutate the caller's input buffer."""
    def body(args):
        (t,) = args
        i0 = _idx(0)
        v = arith.ConstantOp.from_int_and_width(9, 32)
        ins = tensor.InsertOp(v.result, t, [i0.result, i0.result])
        return [i0, v, ins], ins.results[0]
    m = build_module([tensor_ty(2, 2)], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="function argument"):
        apply_bufferization(m)


def test_refuses_scf_for_yielding_a_different_buffer_than_it_carries():
    """A loop must yield the same buffer it carries as an iter_arg -- otherwise the
    dropped-iter_args rewrite would silently pick the wrong buffer's final state."""
    def body(_args):
        ty = tensor_ty(2, 2)
        t_a = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(ty, [0] * 4))
        t_b = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(ty, [1] * 4))
        c0, c1, c2 = _idx(0), _idx(1), _idx(2)
        blk = Block(arg_types=[IndexType(), ty])
        _iv, _carried = blk.args
        blk.add_ops([scf.YieldOp(t_b.result)])   # yields t_b, not the carried %_carried
        loop = scf.ForOp(c0.result, c2.result, c1.result, [t_a.result], Region(blk))
        return [t_a, t_b, c0, c1, c2, loop], loop.results[0]
    m = build_module([], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="different tensor buffer"):
        apply_bufferization(m)


def test_refuses_non_splat_dense_tensor_constant():
    def body(_args):
        t = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 2), [1, 2, 3, 4]))
        return [t], t.results[0]
    m = build_module([], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="splat"):
        apply_bufferization(m)


def test_bufferization_handles_matmul_alongside_a_non_tensor_op_in_the_same_function():
    """A model can mix a tensor op with an ordinary scalar op (e.g. hc.max, which
    lowers to scf.if) in the same function -- exercises the plain-clone path for a
    non-tensor op that owns a nested region, and a mix of tensor and scalar args."""
    from hc_dialect import HCMax

    def body(args):
        a, b, x, y = args
        mm = HCMatmul(operands=[a, b], result_types=[tensor_ty(_M, _N)])
        mx = HCMax(operands=[x, y], result_types=[i32])
        return [mm, mx], mx.results[0]

    m = build_module([tensor_ty(_M, _K), tensor_ty(_K, _N), i32, i32], body)
    lower(m)
    apply_bufferization(m)
    m.verify()
    fn = next(op for blk in m.body.blocks for op in blk.ops if isinstance(op, func.FuncOp))
    a_ty, b_ty, x_ty, y_ty = fn.function_type.inputs
    assert isinstance(a_ty, MemRefType) and isinstance(b_ty, MemRefType)
    assert x_ty == i32 and y_ty == i32          # scalar args pass through _memref_type unchanged
    assert not any(op.name.startswith("tensor.") for op in m.walk())
    assert any(op.name == "scf.if" for op in m.walk())   # hc.max's lowering, untouched by hand


def test_refuses_multi_block_function_body():
    b1 = Block(arg_types=[tensor_ty(2, 2)])
    b1.add_op(func.ReturnOp(b1.args[0]))
    b2 = Block(arg_types=[])
    fn = func.FuncOp("f", ([tensor_ty(2, 2)], [tensor_ty(2, 2)]), region=Region([b1, b2]))
    m = ModuleOp(ops=[fn])
    with pytest.raises(NotImplementedError, match="single-block"):
        apply_bufferization(m)


def test_refuses_an_unhandled_tensor_producing_op():
    def body(_args):
        empty = tensor.EmptyOp([], tensor_ty(2, 2))
        return [empty], empty.results[0]
    m = build_module([], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="unsupported tensor-producing op"):
        apply_bufferization(m)


def test_refuses_tensor_constant_inside_a_nested_region():
    """Tensor constants are handled only in the function's entry block, where
    the lowering places them. A tensor constant inside a loop body is refused."""
    def body(_args):
        c0, c1, c2 = _idx(0), _idx(1), _idx(2)
        blk = Block(arg_types=[IndexType()])
        t = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 2), [5] * 4))
        blk.add_ops([t, scf.YieldOp()])
        loop = scf.ForOp(c0.result, c2.result, c1.result, [], Region(blk))
        dummy = _idx(0)
        return [c0, c1, c2, loop, dummy], dummy.result  # loop's result is unused; only its
        # side effect (running _tensor_constant on a nested block) matters here
    m = build_module([], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="nested region"):
        apply_bufferization(m)


def test_refuses_scf_for_iter_arg_init_tensor_used_again_elsewhere():
    """The tensor passed as a loop's iter_arg init must die there -- if it's also used
    elsewhere, dropping the iter_arg and mutating in place would corrupt that other use.

    Uses the *same* tensor as two separate loops' iter_arg inits (not tensor.insert,
    which has its own independent single-use guard) so this test isolates scf.for's
    own check rather than incidentally passing through the insert guard instead."""
    def body(_args):
        ty = tensor_ty(2, 2)
        t0 = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(ty, [0] * 4))
        c0, c1, c2 = _idx(0), _idx(1), _idx(2)

        blk1 = Block(arg_types=[IndexType(), ty])
        blk1.add_ops([scf.YieldOp(blk1.args[1])])
        loop1 = scf.ForOp(c0.result, c2.result, c1.result, [t0.result], Region(blk1))

        blk2 = Block(arg_types=[IndexType(), ty])
        blk2.add_ops([scf.YieldOp(blk2.args[1])])
        loop2 = scf.ForOp(c0.result, c2.result, c1.result, [t0.result], Region(blk2))
        # t0 is now used by both loops' iter_args -- neither use is exclusive.
        return [t0, c0, c1, c2, loop1, loop2], loop2.results[0]
    m = build_module([], body)
    m.verify()
    with pytest.raises(NotImplementedError, match="used again elsewhere"):
        apply_bufferization(m)


def test_temporary_buffer_not_returned_is_deallocated():
    """Two tensor constants; only one is returned. The other is a temporary this pass
    owns, so it must free it rather than leaking it."""
    def body(_args):
        ty = tensor_ty(2, 2)
        kept = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(ty, [0] * 4))
        temp = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(ty, [7] * 4))
        return [kept, temp], kept.results[0]
    m = build_module([], body)
    # Deliberately skip lower(): with no hc.* ops it's a no-op for LowerHCPattern, but
    # its greedy PatternRewriteWalker driver would still canonicalize away the unused
    # `temp` constant before bufferization ever saw it, defeating the point of this test.
    apply_bufferization(m)
    m.verify()
    (dealloc,) = find_ops(m, "memref.dealloc")
    allocs = find_ops(m, "memref.alloc")
    (ret,) = find_ops(m, "func.return")
    assert dealloc.memref is not ret.operands[0]
    assert {dealloc.memref, ret.operands[0]} == {a.memref for a in allocs}


def test_pipeline_bufferize_flag_wires_in_the_pass():
    """lower(..., bufferize=True) runs apply_bufferization after lowering."""
    m, _, _ = _matmul_module()
    lower(m, bufferize=True)
    m.verify()
    assert not any(op.name.startswith("tensor.") for op in m.walk())
    assert any(op.name == "memref.alloc" for op in m.walk())


def test_apply_bufferization_defaults_on_in_the_pipeline_config():
    """MiddleEndPipelineConfig.apply_bufferization defaults to True, like the
    other pass switches. conftest.lower() always passes the flag explicitly,
    so only this test covers the default."""
    from middle_end.pipeline import MiddleEndPipelineConfig
    assert MiddleEndPipelineConfig().apply_bufferization is True
