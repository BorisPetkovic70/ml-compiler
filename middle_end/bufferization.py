"""Tensor -> memref bufferization of lowered IR. The rewrite table,
ownership rules, and full refusal list are in docs/DESIGN.md Section 4.

`apply_bufferization` rewrites each tensor-using func.func in place:
- tensors become memrefs;
- `tensor.empty` becomes a bare `memref.alloc`;
- splat tensor constants become `memref.alloc` plus a fill loop nest;
- other tensor constants (weights) become a read-only `memref.global`;
- extract/insert become load/store;
- tensor `iter_args` are dropped;
- a write goes into the tensor's own buffer when the function owns it and
  the tensor is dead afterwards (`analysis.is_last_use`); otherwise the
  buffer is first copied and the write goes into the copy;
- a returned buffer the function doesn't own (an argument or a weight) is
  copied, so the caller always owns what it gets back;
- every allocated buffer that isn't returned is deallocated before
  `func.return`.

Copies are placed in the function's top-level block only. When a write
inside a loop body would need one, or the IR isn't one of the shapes the
pass handles, it raises NotImplementedError.

Each function body is rebuilt into a fresh block rather than edited in place,
because `scf.for` can't change its iter_arg count in place. Reading from the
untouched old block also keeps its use lists intact for the `is_last_use`
checks.
"""
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import (
    FunctionType, IndexType, MemRefType, ModuleOp, StringAttr, TensorType, UnitAttr,
)
from xdsl.ir import Block, Operation, Region, SSAValue

from .analysis import is_last_use


def apply_bufferization(module: ModuleOp) -> None:
    """Bufferizes, in place, every func.func in `module` that uses tensors;
    other functions are left untouched."""
    for top_block in module.body.blocks:
        for op in list(top_block.ops):
            if isinstance(op, func.FuncOp) and _uses_tensors(op):
                _Bufferizer(op, module).run()


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def _is_tensor(value: SSAValue) -> bool:
    return isinstance(value.type, TensorType)


def _memref_type(ty):
    """tensor<MxNxT> -> memref<MxNxT>; any other type is returned unchanged."""
    if isinstance(ty, TensorType):
        return MemRefType(ty.element_type, ty.shape)
    return ty


def _values_of(op: Operation):
    """Every SSA value an op mentions or defines, including nested block args."""
    yield from op.operands
    yield from op.results
    for region in op.regions:
        for block in region.blocks:
            yield from block.args


def _has_tensor(op: Operation) -> bool:
    """True if `op` or anything nested inside it touches a tensor value."""
    return any(_is_tensor(v) for nested in op.walk() for v in _values_of(nested))


def _uses_tensors(fn: func.FuncOp) -> bool:
    sig = fn.function_type
    if any(isinstance(t, TensorType) for t in (*sig.inputs, *sig.outputs)):
        return True
    return any(_has_tensor(op) for block in fn.body.blocks for op in block.ops)


def _dim(d) -> int:
    return int(getattr(d, "data", d))


def _fill_ops(buf: SSAValue, tensor_ty: TensorType, value: int) -> list[Operation]:
    """Ops that store `value` into every element of `buf`: a rank-deep scf.for nest."""
    idx = IndexType()
    dims = [_dim(d) for d in tensor_ty.shape]
    # one index constant per distinct value, so equal bounds share one
    consts = {
        v: arith.ConstantOp.from_int_and_width(v, idx) for v in dict.fromkeys([0, 1, *dims])
    }
    c0, c1 = consts[0], consts[1]
    bounds = [consts[d] for d in dims]
    scalar = arith.ConstantOp.from_int_and_width(value, tensor_ty.element_type)

    bodies = [Block(arg_types=[idx]) for _ in dims]
    ivs = [b.args[0] for b in bodies]
    bodies[-1].add_ops([memref.StoreOp.get(scalar.result, buf, ivs), scf.YieldOp()])
    loop = None
    for d in reversed(range(len(dims))):
        if loop is not None:
            bodies[d].add_ops([loop, scf.YieldOp()])
        loop = scf.ForOp(c0.result, bounds[d].result, c1.result, [], Region(bodies[d]))
    return [*consts.values(), scalar, loop]


# -----------------------------------------------------------------------------
#  The pass
# -----------------------------------------------------------------------------
class _Bufferizer:
    """Bufferizes one single-block func.func; `run()` replaces its body and
    updates its function type."""

    def __init__(self, fn: func.FuncOp, module: ModuleOp):
        self.fn = fn
        self.module = module
        # old non-tensor SSA value -> new value; also the value_mapper handed to clone()
        self.vmap: dict[SSAValue, SSAValue] = {}
        # old tensor SSA value -> the memref that now holds it
        self.bufs: dict[SSAValue, SSAValue] = {}
        # memrefs allocated by this pass (safe to mutate / dealloc), in allocation order
        self.owned: list[SSAValue] = []
        self.entry: Block | None = None

    # ---- driver -------------------------------------------------------------
    def run(self) -> None:
        if len(self.fn.body.blocks) != 1:
            raise NotImplementedError("bufferization supports single-block function bodies only")
        old = self.fn.body.block
        new = Block(arg_types=[_memref_type(a.type) for a in old.args])
        self.entry = new

        for old_arg, new_arg in zip(old.args, new.args):
            if _is_tensor(old_arg):
                self.bufs[old_arg] = new_arg      # input buffer: read-only, not owned
            else:
                self.vmap[old_arg] = new_arg

        self._translate(list(old.ops), new)

        ret = new.last_op
        self.fn.function_type = FunctionType.from_lists(
            [a.type for a in new.args], [v.type for v in ret.operands]
        )
        self.fn.body.detach_block(old)
        self.fn.body.add_block(new)
        old.erase(safe_erase=False)

    # ---- op dispatch ----------------------------------------------------------
    def _translate(self, ops: list[Operation], new_block: Block) -> None:
        for op in ops:
            if op.name == "arith.constant" and _is_tensor(op.results[0]):
                self._tensor_constant(op, new_block)
            elif op.name == "tensor.empty":
                self._empty(op, new_block)
            elif op.name == "tensor.extract":
                self._extract(op, new_block)
            elif op.name == "tensor.insert":
                self._insert(op, new_block)
            elif op.name == "scf.for":
                self._for(op, new_block)
            elif op.name == "func.return":
                self._return(op, new_block)
            else:
                self._clone(op, new_block)

    def _clone(self, op: Operation, new_block: Block) -> None:
        if _has_tensor(op):
            raise NotImplementedError(
                f"bufferization: unsupported tensor-producing op {op.name!r} "
                f"(has the pipeline lowered hc.* ops first?)"
            )
        new_block.add_op(op.clone(value_mapper=self.vmap))

    def _buf(self, value: SSAValue) -> SSAValue:
        # KeyError here: a tensor value was reached without first being
        # mapped to a buffer.
        return self.bufs[value]

    def _copy(self, buf: SSAValue, new_block: Block) -> SSAValue:
        """Emits an owned buffer holding a copy of `buf`, and returns it."""
        alloc = memref.AllocOp.get(buf.type.element_type, shape=buf.type.get_shape())
        new_block.add_ops([alloc, memref.CopyOp(buf, alloc.memref)])
        self.owned.append(alloc.memref)
        return alloc.memref

    def _writable(self, value: SSAValue, op: Operation, new_block: Block) -> SSAValue:
        """The buffer `op` may overwrite to produce a new version of the
        tensor `value`: `value`'s own buffer when the function owns it and
        `op` is `value`'s last use, otherwise a copy of it."""
        buf = self._buf(value)
        if buf in self.owned and is_last_use(value, op):
            return buf
        if new_block is not self.entry:
            raise NotImplementedError(
                "bufferization: a write inside a nested region needs a copy of its tensor "
                "(a function argument or weight, or a tensor still needed by a later use "
                "or the next loop iteration); copies are placed in the function's "
                "top-level block only"
            )
        return self._copy(buf, new_block)

    # ---- tensor constant -> alloc + fill, or a read-only global -----------------
    def _tensor_constant(self, op: arith.ConstantOp, new_block: Block) -> None:
        if new_block is not self.entry:
            raise NotImplementedError("bufferization: tensor constants inside a nested region")
        values = list(op.value.get_values())
        ty = op.results[0].type
        if len(set(values)) != 1:
            self._global_constant(op, new_block)
            return
        alloc = memref.AllocOp.get(ty.element_type, shape=[_dim(d) for d in ty.shape])
        buf = alloc.memref
        new_block.add_ops([alloc, *_fill_ops(buf, ty, values[0])])
        self.bufs[op.results[0]] = buf
        self.owned.append(buf)

    def _empty(self, op: Operation, new_block: Block) -> None:
        """A tensor with unspecified contents: an owned buffer, left unfilled."""
        if new_block is not self.entry:
            raise NotImplementedError("bufferization: tensor.empty inside a nested region")
        ty = op.results[0].type
        alloc = memref.AllocOp.get(ty.element_type, shape=[_dim(d) for d in ty.shape])
        new_block.add_op(alloc)
        self.bufs[op.results[0]] = alloc.memref
        self.owned.append(alloc.memref)

    def _global_constant(self, op: arith.ConstantOp, new_block: Block) -> None:
        """A weight: its data goes into a module-level `memref.global constant`,
        read through `memref.get_global`. The buffer is not owned, so it is
        never written to or deallocated."""
        mem_ty = _memref_type(op.results[0].type)
        name = f"__constant_{sum(isinstance(o, memref.GlobalOp) for o in self.module.ops)}"
        glob = memref.GlobalOp.get(StringAttr(name), mem_ty, op.value, constant=UnitAttr())
        self.module.body.block.insert_op_before(glob, self.fn)
        get = memref.GetGlobalOp(name, mem_ty)
        new_block.add_op(get)
        self.bufs[op.results[0]] = get.memref

    # ---- extract / insert -------------------------------------------------------
    def _extract(self, op: Operation, new_block: Block) -> None:
        indices = [self.vmap[i] for i in op.indices]
        load = memref.LoadOp.get(self._buf(op.operands[0]), indices)
        new_block.add_op(load)
        self.vmap[op.results[0]] = load.results[0]

    def _insert(self, op: Operation, new_block: Block) -> None:
        buf = self._writable(op.dest, op, new_block)
        indices = [self.vmap[i] for i in op.indices]
        new_block.add_op(memref.StoreOp.get(self.vmap[op.scalar], buf, indices))
        self.bufs[op.results[0]] = buf          # same buffer, new "version"

    # ---- scf.for: drop tensor iter_args, keep scalar ones ------------------------
    def _for(self, op: scf.ForOp, new_block: Block) -> None:
        old_body = op.body.block
        iv_old, *carried_old = old_body.args
        kept = [i for i, a in enumerate(carried_old) if not _is_tensor(a)]
        dropped = [i for i, a in enumerate(carried_old) if _is_tensor(a)]

        new_body = Block(arg_types=[iv_old.type] + [carried_old[i].type for i in kept])
        self.vmap[iv_old] = new_body.args[0]
        for j, i in enumerate(kept):
            self.vmap[carried_old[i]] = new_body.args[1 + j]
        for i in dropped:
            self.bufs[carried_old[i]] = self._writable(op.iter_args[i], op, new_block)

        *body_ops, terminator = list(old_body.ops)
        self._translate(body_ops, new_body)

        for i in dropped:
            if self.bufs.get(terminator.operands[i]) is not self.bufs[carried_old[i]]:
                raise NotImplementedError(
                    "bufferization: scf.for yields a different tensor buffer than it carries"
                )
        new_body.add_op(scf.YieldOp(*[self.vmap[terminator.operands[i]] for i in kept]))

        new_for = scf.ForOp(
            self.vmap[op.lb], self.vmap[op.ub], self.vmap[op.step],
            [self.vmap[op.iter_args[i]] for i in kept],
            Region(new_body),
        )
        new_block.add_op(new_for)
        for j, i in enumerate(kept):
            self.vmap[op.results[i]] = new_for.results[j]
        for i in dropped:
            self.bufs[op.results[i]] = self.bufs[carried_old[i]]

    # ---- return: hand back buffers, free temporaries -----------------------------
    def _return(self, op: Operation, new_block: Block) -> None:
        operands = []
        for v in op.operands:
            if not _is_tensor(v):
                operands.append(self.vmap[v])
                continue
            buf = self._buf(v)
            if buf not in self.owned:
                # An argument or a weight: the caller frees whatever is returned, so
                # hand back a copy it can own instead.
                buf = self._copy(buf, new_block)
            operands.append(buf)
        # Ownership of a returned buffer passes to the caller; anything else we
        # allocated is a temporary. Freeing at function exit is always safe (a
        # last-use analysis could free earlier -- the extension point for that).
        for buf in self.owned:
            if buf not in operands:
                new_block.add_op(memref.DeallocOp.get(buf))
        new_block.add_op(func.ReturnOp(*operands))
