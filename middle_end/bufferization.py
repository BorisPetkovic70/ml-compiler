"""Hand-written bufferization: tensor (value semantics) -> memref (memory semantics).

Runs on the IR produced by lowering (`tensor.extract`/`tensor.insert`/`scf.for` with
tensor `iter_args`), and rewrites each function that touches tensors:

  * tensor function arguments become memref arguments (read-only inputs)
  * `arith.constant dense<c> : tensor<..>` becomes `memref.alloc` + a store loop nest
    that fills it with `c`
  * `tensor.extract %t[idx]`      -> `memref.load  %m[idx]`
  * `tensor.insert %v into %t[idx]` -> `memref.store %v, %m[idx]`  (in place)
  * tensor `iter_args` on `scf.for` are dropped -- the loop no longer carries the
    tensor, because the stores mutate one buffer; scalar `iter_args` (e.g. the k-loop
    register reduction) are kept
  * a returned tensor becomes a returned memref (allocated here, ownership passes to
    the caller); temporaries that are not returned get a `memref.dealloc`

Turning "every insert yields a new tensor" into "every store mutates one buffer" is
only correct when the old tensor value is dead afterwards. The pass therefore
*proves* that where it matters and refuses (NotImplementedError) otherwise, instead of
silently producing wrong code:

  * a tensor consumed by an insert or a loop `iter_arg` init must have exactly one use;
  * inserting into a function argument would mutate the caller's input -> refused;
  * a loop must yield the same buffer it carries.

Refusing is where a copy (`memref.alloc` + `memref.copy`) would be inserted; that is
deliberately not implemented yet -- none of these trigger on hc.matmul's own nest, but
they keep the pass honest about what it hasn't been shown.

Not yet wired into MiddleEndPipeline, and the interpreter can't execute `memref` ops
yet either -- call `apply_bufferization(module)` directly on an already-lowered module;
there's no way to run the result through the interpreter until that support exists.
The function body is rebuilt (clone-and-translate) rather than edited in place:
`scf.ForOp` can't change its iter_arg/result count in place, and translating from the
untouched old IR keeps the use-count checks above valid.
"""
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import FunctionType, IndexType, MemRefType, ModuleOp, TensorType
from xdsl.ir import Block, Operation, Region, SSAValue


def apply_bufferization(module: ModuleOp) -> None:
    """Bufferize every function in `module` that uses tensors (in place)."""
    for top_block in module.body.blocks:
        for op in list(top_block.ops):
            if isinstance(op, func.FuncOp) and _uses_tensors(op):
                _Bufferizer(op).run()


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
    c0 = arith.ConstantOp.from_int_and_width(0, idx)
    c1 = arith.ConstantOp.from_int_and_width(1, idx)
    bounds = [arith.ConstantOp.from_int_and_width(d, idx) for d in dims]
    scalar = arith.ConstantOp.from_int_and_width(value, tensor_ty.element_type)

    bodies = [Block(arg_types=[idx]) for _ in dims]
    ivs = [b.args[0] for b in bodies]
    bodies[-1].add_ops([memref.StoreOp.get(scalar.result, buf, ivs), scf.YieldOp()])
    loop = None
    for d in reversed(range(len(dims))):
        if loop is not None:
            bodies[d].add_ops([loop, scf.YieldOp()])
        loop = scf.ForOp(c0.result, bounds[d].result, c1.result, [], Region(bodies[d]))
    return [c0, c1, *bounds, scalar, loop]


# -----------------------------------------------------------------------------
#  The pass
# -----------------------------------------------------------------------------
class _Bufferizer:
    def __init__(self, fn: func.FuncOp):
        self.fn = fn
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
        buf = self.bufs.get(value)
        if buf is None:
            # Defensive fallback, not currently reachable: every tensor-producing op is
            # either registered here (a function argument, a splat constant, an insert,
            # or a dropped scf.for iter_arg) or refused earlier by _clone (line ~164)
            # before its result can reach _buf. Left uncovered rather than fabricating
            # an artificial call path, per the project's coverage discipline.
            raise NotImplementedError(
                f"bufferization: tensor value has no buffer (produced by "
                f"{getattr(value.owner, 'name', 'a block argument')!r})"
            )
        return buf

    # ---- tensor constant -> alloc + fill ---------------------------------------
    def _tensor_constant(self, op: arith.ConstantOp, new_block: Block) -> None:
        if new_block is not self.entry:
            raise NotImplementedError("bufferization: tensor constants inside a nested region")
        values = list(op.value.get_values())
        if len(set(values)) != 1:
            raise NotImplementedError("bufferization: only splat dense tensor constants")
        ty = op.results[0].type
        alloc = memref.AllocOp.get(ty.element_type, shape=[_dim(d) for d in ty.shape])
        buf = alloc.memref
        new_block.add_ops([alloc, *_fill_ops(buf, ty, values[0])])
        self.bufs[op.results[0]] = buf
        self.owned.append(buf)

    # ---- extract / insert -------------------------------------------------------
    def _extract(self, op: Operation, new_block: Block) -> None:
        indices = [self.vmap[i] for i in op.indices]
        load = memref.LoadOp.get(self._buf(op.operands[0]), indices)
        new_block.add_op(load)
        self.vmap[op.results[0]] = load.results[0]

    def _insert(self, op: Operation, new_block: Block) -> None:
        dest = op.dest
        buf = self._buf(dest)
        if buf not in self.owned:
            raise NotImplementedError(
                "bufferization: tensor.insert into a function argument would mutate the "
                "caller's input (needs a copy)"
            )
        if not dest.has_one_use():
            raise NotImplementedError(
                "bufferization: the tensor being inserted into is used again elsewhere, so "
                "an in-place store would change what that other use sees (needs a copy)"
            )
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
            init = op.iter_args[i]
            if not init.has_one_use():
                raise NotImplementedError(
                    "bufferization: a tensor carried by scf.for is used again elsewhere "
                    "(needs a copy)"
                )
            self.bufs[carried_old[i]] = self._buf(init)

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
        operands = [self._buf(v) if _is_tensor(v) else self.vmap[v] for v in op.operands]
        # Ownership of a returned buffer passes to the caller; anything else we
        # allocated is a temporary. Freeing at function exit is always safe (a
        # last-use analysis could free earlier -- the extension point for that).
        for buf in self.owned:
            if buf not in operands:
                new_block.add_op(memref.DeallocOp.get(buf))
        new_block.add_op(func.ReturnOp(*operands))
