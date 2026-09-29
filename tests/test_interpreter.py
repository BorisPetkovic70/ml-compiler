"""Operator semantics: each op gives the same interpreter result before and
after lowering (and after bufferization, for matmul), plus unit tests of the
interpreter's own tensor, memref, and scf.for handling.
"""
import pytest

from xdsl.ir import Block, Region
from xdsl.dialects import arith, memref, scf, tensor
from xdsl.dialects.builtin import i32, IndexType, DenseIntOrFPElementsAttr
from conftest import build_module, const_i32, const_vec, tensor_ty, run, lower
from hc_dialect import (
    HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,
    HCAddVec, HCSubVec, HCMulVec, HCMulVecVec, HCReluVec, HCMatmul,
    HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor,
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
    """The scalar is a function argument rather than a constant, so the
    lowering must broadcast it at runtime."""
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
    """y = relu(a * x + b) with scalar a and vectors x, b: multiple inputs,
    mixed scalar and vector, and a mul/add/relu chain."""
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


# =============================================================================
#  Tensors, hc.matmul and the general scf.for executor 
# =============================================================================

def _idx_const(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(value, IndexType())


def _matmul_module(m: int, k: int, n: int, batch: int | None = None):
    a_ty = tensor_ty(batch, m, k) if batch else tensor_ty(m, k)
    b_ty = tensor_ty(batch, k, n) if batch else tensor_ty(k, n)
    res_ty = tensor_ty(batch, m, n) if batch else tensor_ty(m, n)

    def body(args):
        a, b = args
        op = HCMatmul(operands=[a, b], result_types=[res_ty])
        return [op], op.results[0]
    return build_module([a_ty, b_ty], body)


@pytest.mark.parametrize("batch,m,k,n", [
    (None, 2, 3, 4), (None, 1, 1, 1), (None, 3, 1, 2), (None, 1, 5, 1), (None, 4, 4, 4),
    (2, 2, 3, 4), (3, 1, 2, 2),
])
@pytest.mark.parametrize("fold,dce,bufferize", [
    (False, False, False), (True, True, False), (False, False, True), (True, True, True),
])
def test_matmul_semantics_match_numpy_before_and_after_lowering(batch, m, k, n, fold, dce, bufferize):
    """The native hc.matmul, its lowered tensor nest, and (with bufferize=True)
    its memref nest all equal np.matmul. Cases cover signed values, distinct
    dims, plain and batched shapes, and runs with and without folding/DCE."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=m * 100 + k * 10 + n + (batch or 0) * 1000)
    a_shape = (batch, m, k) if batch else (m, k)
    b_shape = (batch, k, n) if batch else (k, n)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    expected = np.matmul(a, b).tolist()
    args = [a.tolist(), b.tolist()]

    module = _matmul_module(m, k, n, batch)
    assert run(module, args) == expected          # native hc.matmul
    lower(module, fold=fold, dce=dce, bufferize=bufferize)
    assert run(module, args) == expected          # lowered (and maybe bufferized) IR


def _shape_of(vals) -> list[int]:
    shape = []
    v = vals
    while isinstance(v, list):
        shape.append(len(v))
        v = v[0] if v else None
    return shape


def _tensor_binop_module(op_cls, a_vals, b_vals):
    def body(args):
        a, b = args
        op = op_cls(operands=[a, b], result_types=[a.type])
        return [op], op.results[0]
    ty = tensor_ty(*_shape_of(a_vals))
    m = build_module([ty, ty], body)
    return m, a_vals, b_vals


@pytest.mark.parametrize("op_cls,a,b,expected", [
    (HCAddTensor, [[1, 2], [3, 4]], [[10, 20], [30, 40]], [[11, 22], [33, 44]]),
    (HCSubTensor, [[10, 20], [30, 40]], [[1, 2], [3, 4]], [[9, 18], [27, 36]]),
    (HCMulTensor, [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[2, 6], [12, 20]]),
    (HCAddTensor, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
     [[[10, 20], [30, 40]], [[50, 60], [70, 80]]],
     [[[11, 22], [33, 44]], [[55, 66], [77, 88]]]),
    (HCMulTensor, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
     [[[2, 3], [4, 5]], [[2, 3], [4, 5]]],
     [[[2, 6], [12, 20]], [[10, 18], [28, 40]]]),
])
def test_tensor_binop_semantics_before_and_after_lowering(op_cls, a, b, expected):
    """Tensor binops on 2D and batched 3D nested lists, before and after
    lowering. Catches an _elt_binop that concatenates rows instead of recursing
    into them."""
    m, a_vals, b_vals = _tensor_binop_module(op_cls, a, b)
    args = [a_vals, b_vals]
    assert run(m, args) == expected
    lower(m)
    assert run(m, args) == expected


def test_relu_tensor_semantics_before_and_after_lowering():
    def body(args):
        (x,) = args
        r = HCReluTensor(operands=[x], result_types=[x.type])
        return [r], r.results[0]
    m = build_module([tensor_ty(2, 2)], body)
    x = [[-1, 2], [3, -4]]
    expected = [[0, 2], [3, 0]]
    assert run(m, args=[x]) == expected
    lower(m)
    assert run(m, args=[x]) == expected


def test_relu_tensor_semantics_before_and_after_lowering_batched():
    def body(args):
        (x,) = args
        r = HCReluTensor(operands=[x], result_types=[x.type])
        return [r], r.results[0]
    m = build_module([tensor_ty(2, 2, 2)], body)
    x = [[[-1, 2], [3, -4]], [[5, -6], [-7, 8]]]
    expected = [[[0, 2], [3, 0]], [[5, 0], [0, 8]]]
    assert run(m, args=[x]) == expected
    lower(m)
    assert run(m, args=[x]) == expected


def _matmul_then_relu_module(m: int, k: int, n: int, batch: int | None = None):
    a_ty = tensor_ty(batch, m, k) if batch else tensor_ty(m, k)
    b_ty = tensor_ty(batch, k, n) if batch else tensor_ty(k, n)
    res_ty = tensor_ty(batch, m, n) if batch else tensor_ty(m, n)

    def body(args):
        a, b = args
        mm = HCMatmul(operands=[a, b], result_types=[res_ty])
        relu = HCReluTensor(operands=[mm.results[0]], result_types=[res_ty])
        return [mm, relu], relu.results[0]
    return build_module([a_ty, b_ty], body)


@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("fold,dce,bufferize", [
    (False, False, False), (True, True, False), (False, False, True), (True, True, True),
])
def test_chained_matmul_then_relu_tensor_semantics_match_numpy(batch, fold, dce, bufferize):
    """hc.matmul feeding hc.relu_tensor equals max(A @ B, 0), plain and batched.
    This checks that the intermediate result is threaded correctly from one
    nest into the next."""
    np = pytest.importorskip("numpy")
    m, k, n = 2, 3, 4
    rng = np.random.default_rng(seed=42 + (batch or 0))
    a_shape = (batch, m, k) if batch else (m, k)
    b_shape = (batch, k, n) if batch else (k, n)
    a = rng.integers(-5, 6, size=a_shape)
    b = rng.integers(-5, 6, size=b_shape)
    expected = np.maximum(np.matmul(a, b), 0).tolist()
    args = [a.tolist(), b.tolist()]

    module = _matmul_then_relu_module(m, k, n, batch)
    assert run(module, args) == expected                                  # native
    lower(module, fold=fold, dce=dce, bufferize=bufferize)
    assert run(module, args) == expected                                  # lowered/bufferized


def test_dense_tensor_constant_is_read_row_major():
    """A transposed reshape would read [1,0] as 4 and [0,2] as 3."""
    def body(_args):
        t = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 3), [1, 2, 3, 4, 5, 6]))
        i0, i1, i2 = _idx_const(0), _idx_const(1), _idx_const(2)
        x = tensor.ExtractOp(t.result, [i1.result, i0.result], i32)   # row 1, col 0
        y = tensor.ExtractOp(t.result, [i0.result, i2.result], i32)   # row 0, col 2
        ten = const_i32(10)
        scaled = arith.MuliOp(x.result, ten.result)
        out = arith.AddiOp(scaled.result, y.result)
        return [t, i0, i1, i2, x, y, ten, scaled, out], out.result
    m = build_module([], body)
    m.verify()
    assert run(m) == 4 * 10 + 3


def test_tensor_insert_has_value_semantics_source_unchanged():
    """insert must return a new tensor. If it mutated its source, reading the
    original afterwards would give 9 (-> 909) instead of 2 (-> 209)."""
    def body(_args):
        t0 = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 2), [1, 2, 3, 4]))
        i0, i1 = _idx_const(0), _idx_const(1)
        nine = const_i32(9)
        t1 = tensor.InsertOp(nine.result, t0.result, [i0.result, i1.result])
        old = tensor.ExtractOp(t0.result, [i0.result, i1.result], i32)
        new = tensor.ExtractOp(t1.result, [i0.result, i1.result], i32)
        hundred = const_i32(100)
        scaled = arith.MuliOp(old.result, hundred.result)
        out = arith.AddiOp(scaled.result, new.result)
        return [t0, i0, i1, nine, t1, old, new, hundred, scaled, out], out.result
    m = build_module([], body)
    m.verify()
    assert run(m) == 2 * 100 + 9


def _extract_at(row: int, col: int):
    def body(_args):
        t = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 2), [1, 2, 3, 4]))
        r, c = _idx_const(row), _idx_const(col)
        x = tensor.ExtractOp(t.result, [r.result, c.result], i32)
        return [t, r, c, x], x.result
    return build_module([], body)


@pytest.mark.parametrize("row,col", [(2, 0), (0, 2), (0, -1), (-1, 0)])
def test_tensor_extract_out_of_bounds_raises(row, col):
    """Includes negative indices, which Python lists would silently wrap."""
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(_extract_at(row, col))


def test_tensor_insert_out_of_bounds_raises():
    def body(_args):
        t = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_ty(2, 2), [1, 2, 3, 4]))
        r, c = _idx_const(2), _idx_const(0)
        v = const_i32(7)
        ins = tensor.InsertOp(v.result, t.result, [r.result, c.result])
        return [t, r, c, v, ins], ins.result
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(build_module([], body))


# ---- general scf.for executor (not just the hc.pow shape) --------------------

def _for_sum_module(lb: int, ub: int, step: int):
    """sum of the induction variable over range(lb, ub, step)."""
    def body(_args):
        c_lb, c_ub, c_step = _idx_const(lb), _idx_const(ub), _idx_const(step)
        init = const_i32(0)
        blk = Block(arg_types=[IndexType(), i32])
        iv, acc = blk.args
        cast = arith.IndexCastOp(iv, i32)
        add = arith.AddiOp(acc, cast.result)
        blk.add_ops([cast, add, scf.YieldOp(add.result)])
        loop = scf.ForOp(c_lb.result, c_ub.result, c_step.result, [init.result], Region(blk))
        return [c_lb, c_ub, c_step, init, loop], loop.results[0]
    m = build_module([], body)
    m.verify()
    return m


@pytest.mark.parametrize("lb,ub,step", [
    (0, 5, 1),   # plain
    (1, 10, 3),  # non-unit lower bound and step: hc.pow's matcher required lb=0, step=1
    (0, 0, 1),   # zero-trip: result is the initial value
    (5, 2, 1),   # lb > ub: also zero-trip
])
def test_scf_for_binds_induction_variable_and_step(lb, ub, step):
    assert run(_for_sum_module(lb, ub, step)) == sum(range(lb, ub, step))


def test_scf_for_rejects_non_positive_step():
    with pytest.raises(RuntimeError, match="step must be positive"):
        run(_for_sum_module(0, 5, 0))


@pytest.mark.parametrize("n", [0, 1, 6])
def test_scf_for_threads_multiple_iter_args_in_yield_order(n):
    """Fibonacci carries two values and yields them in swapped order, so a mix-up
    between iter_args / yield operands / results changes the answer."""
    def body(_args):
        c0, c1, cn = _idx_const(0), _idx_const(1), _idx_const(n)
        a0, b0 = const_i32(0), const_i32(1)
        blk = Block(arg_types=[IndexType(), i32, i32])
        _iv, a, b = blk.args
        s = arith.AddiOp(a, b)
        blk.add_ops([s, scf.YieldOp(b, s.result)])       # (a, b) <- (b, a + b)
        loop = scf.ForOp(c0.result, cn.result, c1.result, [a0.result, b0.result], Region(blk))
        return [c0, c1, cn, a0, b0, loop], loop.results[0]
    m = build_module([], body)
    m.verify()
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    assert run(m) == a


@pytest.mark.parametrize("base,exp", [
    (2, 0),    # zero-trip loop -> 1
    (2, 1),    # one trip
    (3, 4),
    (-2, 3),   # negative base
    (7, 2),
])
def test_pow_semantics_regression_through_general_scf_for(base, exp):
    """hc.pow gives the same result before lowering and after, when its
    multiply loop runs on the general scf.for executor."""
    def body(_args):
        b, e = const_i32(base), const_i32(exp)
        op = HCPow(operands=[b.result, e.result], result_types=[b.result.type])
        return [b, e, op], op.results[0]
    m = build_module([], body)
    assert run(m) == base ** exp
    lower(m)
    assert run(m) == base ** exp


# ---- guards against malformed input (each raise has a negative test) ----------

def test_matmul_rejects_runtime_inner_dimension_mismatch():
    """Declared types are 2x3 @ 3x4, but the values passed in disagree."""
    m = _matmul_module(2, 3, 4)
    a_bad = [[1, 2], [3, 4]]                 # 2x2, not 2x3
    b = [[1, 2, 3, 4]] * 3
    with pytest.raises(RuntimeError, match="inner dimensions"):
        run(m, [a_bad, b])


def test_reshape_rejects_wrong_element_count():
    from simulator.interpreter import _reshape
    with pytest.raises(RuntimeError, match="Cannot reshape"):
        _reshape([1, 2, 3], [2, 2])


def test_scf_for_rejects_yield_count_mismatch():
    """One iter_arg / one result, but the body yields two values (unverified IR:
    zip() would otherwise silently drop the extra one)."""
    def body(_args):
        c0, c1, c3 = _idx_const(0), _idx_const(1), _idx_const(3)
        init = const_i32(0)
        blk = Block(arg_types=[IndexType(), i32])
        _iv, acc = blk.args
        blk.add_ops([scf.YieldOp(acc, acc)])
        loop = scf.ForOp(c0.result, c3.result, c1.result, [init.result], Region(blk))
        return [c0, c1, c3, init, loop], loop.results[0]
    with pytest.raises(RuntimeError, match="yields 2 values"):
        run(build_module([], body))


# ---- memref: memory semantics (alloc/load/store/dealloc), the opposite of tensor's
#      value semantics -- exercised directly rather than only inferred from matmul ----

def test_memref_store_is_visible_through_a_different_aliasing_value():
    """The crux of memory semantics: two separate SSA values that both load from the
    same buffer must both see a write made through it -- the opposite of
    tensor.insert, where every SSA value is its own independent snapshot."""
    def body(_args):
        alloc = memref.AllocOp.get(i32, shape=[2, 2])
        i0, i1 = _idx_const(0), _idx_const(1)
        seven = const_i32(7)
        store = memref.StoreOp.get(seven.result, alloc.memref, [i0.result, i1.result])
        load_a = memref.LoadOp.get(alloc.memref, [i0.result, i1.result])
        load_b = memref.LoadOp.get(alloc.memref, [i0.result, i1.result])
        out = arith.AddiOp(load_a.results[0], load_b.results[0])
        return [alloc, i0, i1, seven, store, load_a, load_b, out], out.result
    m = build_module([], body)
    m.verify()
    assert run(m) == 14  # both loads see the stored 7, not an uninitialized/stale value


def _memref_load_at(row: int, col: int):
    def body(_args):
        alloc = memref.AllocOp.get(i32, shape=[2, 2])
        i0, i1 = _idx_const(0), _idx_const(0)
        seven = const_i32(7)
        store = memref.StoreOp.get(seven.result, alloc.memref, [i0.result, i1.result])
        r, c = _idx_const(row), _idx_const(col)
        load = memref.LoadOp.get(alloc.memref, [r.result, c.result])
        return [alloc, i0, i1, seven, store, r, c, load], load.results[0]
    return build_module([], body)


@pytest.mark.parametrize("row,col", [(2, 0), (0, 2), (0, -1), (-1, 0)])
def test_memref_load_out_of_bounds_raises(row, col):
    """Includes negative indices, which Python lists would silently wrap."""
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(_memref_load_at(row, col))


def test_memref_store_out_of_bounds_raises():
    def body(_args):
        alloc = memref.AllocOp.get(i32, shape=[2, 2])
        r, c = _idx_const(2), _idx_const(0)
        v = const_i32(7)
        store = memref.StoreOp.get(v.result, alloc.memref, [r.result, c.result])
        zero = const_i32(0)
        return [alloc, r, c, v, store, zero], zero.result
    with pytest.raises(RuntimeError, match="out of bounds"):
        run(build_module([], body))


def test_memref_use_after_dealloc_raises():
    """dealloc deletes the buffer's binding rather than adding a bespoke check, so a
    later use fails exactly the way any other missing-value interpreter bug would."""
    def body(_args):
        alloc = memref.AllocOp.get(i32, shape=[2, 2])
        i0 = _idx_const(0)
        dealloc = memref.DeallocOp.get(alloc.memref)
        load = memref.LoadOp.get(alloc.memref, [i0.result, i0.result])
        return [alloc, i0, dealloc, load], load.results[0]
    with pytest.raises(KeyError):
        run(build_module([], body))


def test_memref_alloc_fills_with_none_so_an_unwritten_cell_fails_loudly():
    """alloc deliberately fills with None, not 0: a cell nothing ever writes to must
    fail the moment it's used in arithmetic, rather than silently computing with 0."""
    def body(_args):
        alloc = memref.AllocOp.get(i32, shape=[2, 2])
        i0, i1 = _idx_const(0), _idx_const(1)
        seven = const_i32(7)
        store = memref.StoreOp.get(seven.result, alloc.memref, [i0.result, i0.result])
        load = memref.LoadOp.get(alloc.memref, [i0.result, i1.result])  # [0,1] never written
        out = arith.AddiOp(load.results[0], seven.result)
        return [alloc, i0, i1, seven, store, load, out], out.result
    with pytest.raises(TypeError):
        run(build_module([], body))
