"""Pure-Python interpreter for the hc, arith, vector, tensor, memref, and scf
ops this pipeline emits. It is the test suite's correctness oracle
(docs/DESIGN.md Section 7).

Values: scalars and `index` values are Python ints; vectors are flat lists;
tensors and memrefs are row-major nested lists. Tensor ops return copies;
memref ops mutate in place. An unimplemented op raises RuntimeError.

The `hc` binary ops broadcast their operands (`_broadcast_binop`). The
`arith` ops take operands of one type and do not (`_elt_binop`).

Simplifications:
- Ints are unbounded, so i32 overflow never wraps.
- `arith.cmpi` supports only `sgt` and `slt`.
- `scf.if` returns the taken branch's yielded value without executing the
  branch's other ops. This is correct for the `hc.max`/`hc.min` lowering,
  whose branches yield values defined outside the `scf.if`.
"""
from typing import Any
from xdsl.ir import Operation, SSAValue
from xdsl.dialects import func
from xdsl.dialects.builtin import MemRefType, TensorType, VectorType

# -----------------------------
# Helpers to read constants
# -----------------------------

def _int_from_attr(attr: Any) -> int | None:
    """
    Extract an int from common xdsl attribute shapes (IntegerAttr/IntAttr, etc.).
    """
    if attr is None:
        return None

    # Often: attr.value is IntAttr; or attr.data is int; or nested.
    for k in ("data", "value"):
        if hasattr(attr, k):
            inner = getattr(attr, k)
            # inner might again be an attr
            for k2 in ("data", "value"):
                if hasattr(inner, k2):
                    try:
                        return int(getattr(inner, k2))
                    except Exception:
                        pass
            try:
                return int(inner)
            except Exception:
                pass

    try:
        return int(attr)
    except Exception:
        return None


def _const_value(op: Operation) -> int | None:
    """
    arith.constant -> python int
    """
    if op.name != "arith.constant":
        return None
    v = getattr(op, "value", None)
    return _int_from_attr(v)


def _dense_values(op: Operation) -> list[int] | None:
    """Flat list[int] of a dense `arith.constant` (vector or tensor), else None."""
    if op.name != "arith.constant":
        return None
    v = getattr(op, "value", None)
    if v is None or not hasattr(v, "get_values"):
        return None
    try:
        return list(v.get_values())
    except Exception:
        return None


def _elt_binop(a, b, f):
    """Applies f element-wise over nested lists of any depth, broadcasting a
    scalar against a list."""
    if isinstance(a, list) and isinstance(b, list):
        return [_elt_binop(x, y, f) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_elt_binop(x, b, f) for x in a]
    if isinstance(b, list):
        return [_elt_binop(a, y, f) for y in b]
    return f(a, b)


def _depth(x) -> int:
    """Nesting depth of a value: 0 for a scalar, 1 for a flat list, and so
    on."""
    return 1 + _depth(x[0]) if isinstance(x, list) else 0


def _broadcast_binop(a, b, f):
    """Applies f element-wise with NumPy broadcasting, written independently
    of the lowering.

    The shallower operand is paired, whole, with each element of the deeper
    one, which aligns the two at their last dim. At equal depth, a list of
    length 1 is repeated to the other's length. Raises RuntimeError when two
    lengths differ and neither is 1.
    """
    depth_a, depth_b = _depth(a), _depth(b)
    if depth_a == depth_b == 0:
        return f(a, b)
    if depth_a < depth_b:
        return [_broadcast_binop(a, y, f) for y in b]
    if depth_b < depth_a:
        return [_broadcast_binop(x, b, f) for x in a]
    if len(a) == 1:
        a = a * len(b)
    if len(b) == 1:
        b = b * len(a)
    if len(a) != len(b):
        raise RuntimeError(f"Cannot broadcast dims of size {len(a)} and {len(b)}")
    return [_broadcast_binop(x, y, f) for x, y in zip(a, b)]


def _vec_len(vec_type: VectorType) -> int:
    n = 1
    for d in vec_type.shape:
        n *= int(getattr(d, "data", d))
    return n


# -----------------------------
# Tensor helpers
#
# A tensor value is a nested Python list (row-major): a 2-D tensor is a list of
# rows. Tensors have value semantics, so nothing here mutates its input.
# -----------------------------

def _tensor_shape(ty: TensorType | MemRefType) -> list[int]:
    return [int(getattr(d, "data", d)) for d in ty.shape]


def _reshape(flat: list, shape: list[int]) -> list:
    """Row-major flat list -> nested lists of the given shape."""
    total = 1
    for d in shape:
        total *= d
    if len(flat) != total:
        raise RuntimeError(f"Cannot reshape {len(flat)} values into shape {shape}")
    if len(shape) <= 1:
        return list(flat)
    step = total // shape[0]
    return [_reshape(flat[r * step:(r + 1) * step], shape[1:]) for r in range(shape[0])]


def _check_index(t: list, i: int) -> None:
    # Python would silently wrap a negative index, so bound-check explicitly.
    if not 0 <= i < len(t):
        raise RuntimeError(f"Tensor index {i} out of bounds for dimension of size {len(t)}")


def _tensor_extract(t: list, indices: list[int]):
    for i in indices:
        _check_index(t, i)
        t = t[i]
    return t


def _tensor_insert(t: list, indices: list[int], value) -> list:
    """Copy of `t` with the element at `indices` replaced (`t` is left intact)."""
    i, rest = indices[0], indices[1:]
    _check_index(t, i)
    new = list(t)
    new[i] = _tensor_insert(t[i], rest, value) if rest else value
    return new


# -----------------------------
# Memref helpers
#
# A memref value uses the same nested-list representation as a tensor, but
# memref.store mutates it in place, so every SSA value aliasing the buffer
# observes the write.
# -----------------------------

def _memref_store(buf: list, indices: list[int], value) -> None:
    """Writes `value` into `buf` at `indices`, in place, after bounds-checking
    every index."""
    for i in indices[:-1]:
        _check_index(buf, i)
        buf = buf[i]
    _check_index(buf, indices[-1])
    buf[indices[-1]] = value


def _memref_copy(src: list, dst: list) -> None:
    """Copies every element of `src` into `dst`, in place."""
    for i, item in enumerate(src):
        if isinstance(item, list):
            _memref_copy(item, dst[i])
        else:
            dst[i] = item


def _matmul(a: list, b: list) -> list:
    """Reference (MxK) @ (KxN) -> (MxN), deliberately independent of the
    lowering. Operands with leading batch dims are multiplied one pair of
    slices at a time."""
    if _depth(a) > 2:
        if len(a) != len(b):
            raise RuntimeError("matmul: batch dimensions do not agree")
        return [_matmul(a_slice, b_slice) for a_slice, b_slice in zip(a, b)]
    k = len(b)
    n = len(b[0]) if k else 0
    if any(len(row) != k for row in a):
        raise RuntimeError("matmul: inner dimensions do not agree")
    return [[sum(row[x] * b[x][j] for x in range(k)) for j in range(n)] for row in a]

# -----------------------------
# Interpreter
# -----------------------------

class Interpreter:
    """Executes one function. `env` maps id(SSAValue) to the value's Python
    representation."""

    def __init__(self):
        # SSAValue is not always hashable across versions -> use id()
        self.env: dict[int, object] = {}
        # module-level memref.global ops, by symbol name
        self.globals: dict[str, Operation] = {}

    def _get(self, v: SSAValue):
        return self.env[id(v)]

    def _set(self, v: SSAValue, value) -> None:
        # print(f"Set SSAValue: {v} = {value}")
        self.env[id(v)] = value

    def run_block(self, block):
        """Executes `block`'s ops in order. Returns the `func.return` operand,
        or the list of `scf.yield` operands for a loop body. Raises
        RuntimeError for an unsupported op or a block without a terminator."""
        for op in list(block.ops):
            name = op.name

            # --- constants ---
            if name == "arith.constant":
                dense = _dense_values(op)
                if dense is not None:
                    ty = op.results[0].type
                    if isinstance(ty, TensorType):
                        dense = _reshape(dense, _tensor_shape(ty))
                    self._set(op.results[0], dense)
                    continue
                val = _const_value(op)
                if val is None:
                    raise RuntimeError(f"Couldn't read constant value from op: {op}")
                self._set(op.results[0], val)
                continue

            # --- high-level HC ops ---
            if name == "hc.add":
                a, b = op.operands
                self._set(op.results[0], _broadcast_binop(self._get(a), self._get(b), lambda p, q: p + q))
                continue

            if name == "hc.sub":
                a, b = op.operands
                self._set(op.results[0], _broadcast_binop(self._get(a), self._get(b), lambda p, q: p - q))
                continue

            if name == "hc.mul":
                a, b = op.operands
                self._set(op.results[0], _broadcast_binop(self._get(a), self._get(b), lambda p, q: p * q))
                continue

            if name == "hc.relu":
                (x,) = op.operands
                self._set(op.results[0], _elt_binop(self._get(x), 0, max))
                continue

            if name == "hc.pow":
                a, b = op.operands
                self._set(op.results[0], _broadcast_binop(self._get(a), self._get(b), lambda p, q: p ** q))
                continue

            if name == "hc.max":
                a, b = op.operands
                self._set(op.results[0], _broadcast_binop(self._get(a), self._get(b), max))
                continue

            if name == "hc.min":
                a, b = op.operands
                self._set(op.results[0], _broadcast_binop(self._get(a), self._get(b), min))
                continue

            if name == "hc.matmul":
                a, b = op.operands
                self._set(op.results[0], _matmul(self._get(a), self._get(b)))
                continue

            # --- tensor ops (value semantics: insert returns a new tensor) ---
            if name == "tensor.empty":
                # Unspecified contents are None, so reading an element that was
                # never inserted fails on the first arithmetic that touches it.
                shape = _tensor_shape(op.results[0].type)
                total = 1
                for d in shape:
                    total *= d
                self._set(op.results[0], _reshape([None] * total, shape))
                continue

            if name == "tensor.extract":
                t = self._get(op.operands[0])
                idx = [self._get(i) for i in op.indices]
                self._set(op.results[0], _tensor_extract(t, idx))
                continue

            if name == "tensor.insert":
                idx = [self._get(i) for i in op.indices]
                self._set(
                    op.results[0],
                    _tensor_insert(self._get(op.dest), idx, self._get(op.scalar)),
                )
                continue

            # --- memref ops (memory semantics: store mutates its buffer in place) ---
            if name == "memref.alloc":
                shape = _tensor_shape(op.results[0].type)
                total = 1
                for d in shape:
                    total *= d
                # Filled with None, not 0: a cell that is read before anything stored
                # into it fails loudly on the first arithmetic that touches it, instead
                # of silently computing with a plausible-looking 0.
                self._set(op.results[0], _reshape([None] * total, shape))
                continue

            if name == "memref.load":
                buf = self._get(op.memref)
                idx = [self._get(i) for i in op.indices]
                self._set(op.results[0], _tensor_extract(buf, idx))
                continue

            if name == "memref.store":
                buf = self._get(op.memref)
                idx = [self._get(i) for i in op.indices]
                _memref_store(buf, idx, self._get(op.value))
                continue

            if name == "memref.get_global":
                # A fresh nested list on every read: the global's data never changes.
                glob = self.globals[op.name_.root_reference.data]
                shape = _tensor_shape(glob.type)
                self._set(op.results[0], _reshape(list(glob.initial_value.get_values()), shape))
                continue

            if name == "memref.copy":
                _memref_copy(self._get(op.source), self._get(op.destination))
                continue

            if name == "memref.dealloc":
                # No bespoke use-after-free check: deleting the binding means any later
                # use of this buffer raises the same KeyError any other missing-value
                # bug already would.
                del self.env[id(op.memref)]
                continue

            # --- lowered arith ops (so you can run after lowering too) ---
            if name == "arith.addi":
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), lambda p, q: p + q))
                continue

            if name == "arith.subi":
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), lambda p, q: p - q))
                continue

            if name == "arith.muli":
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), lambda p, q: p * q))
                continue

            if name == "arith.maxsi":
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), max))
                continue

            if name == "arith.minsi":
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), min))
                continue

            if name == "arith.index_cast":
                # index vs i32 is not distinguished in this Python-int interpreter.
                (a,) = op.operands
                self._set(op.results[0], self._get(a))
                continue

            if name == "vector.broadcast":
                (src,) = op.operands
                n = _vec_len(op.results[0].type)
                self._set(op.results[0], [self._get(src)] * n)
                continue

            if name == "arith.cmpi":
                a, b = op.operands
                lhs = self._get(a)
                rhs = self._get(b)

                pred = op.predicate.value.data

                # arith.cmpi predicate encoding: slt == 2, sgt == 4
                if pred == 4:  # signed greater-than
                    result = 1 if lhs > rhs else 0
                elif pred == 2:  # signed less-than
                    result = 1 if lhs < rhs else 0
                else:
                    raise RuntimeError(
                        f"Only sgt and slt supported in interpreter, got predicate: {pred}\nOp: {op}"
                    )
                self._set(op.results[0], result)
                continue

            if name == "scf.for":
                lb, ub, step = (self._get(v) for v in (op.lb, op.ub, op.step))
                if step <= 0:
                    raise RuntimeError(f"scf.for step must be positive, got {step}")
                carried = [self._get(v) for v in op.iter_args]
                body = op.body.block
                iv_arg, *carried_args = body.args
                for iv in range(lb, ub, step):
                    self._set(iv_arg, iv)
                    for arg, val in zip(carried_args, carried):
                        self._set(arg, val)
                    carried = self.run_block(body)  # values scf.yield hands back
                if len(carried) != len(op.results):
                    raise RuntimeError(
                        f"scf.for yields {len(carried)} values, expected {len(op.results)}"
                    )
                for res, val in zip(op.results, carried):
                    self._set(res, val)
                continue

            if name == "scf.yield":
                return [self._get(v) for v in op.operands]

            if name == "scf.if":
                # Treat non-zero as True
                cond = self._get(op.operands[0]) != 0

                then_region = op.regions[0]
                else_region = op.regions[1] if len(op.regions) > 1 else None

                if cond:
                    yielded = self._eval_scf_if_region_yield(then_region)
                else:
                    if else_region is None:
                        raise RuntimeError("scf.if has no else region but condition is false")
                    yielded = self._eval_scf_if_region_yield(else_region)

                # Assume scf.if returns exactly one value
                self._set(op.results[0], yielded)
                continue

            # --- return ---
            if name == "func.return":
                if len(op.operands) != 1:
                    raise RuntimeError(f"Expected func.return with 1 operand, got: {op}")
                return self._get(op.operands[0])

            raise RuntimeError(f"Unsupported op in interpreter: {name}\nOp: {op}")

        raise RuntimeError("Block ended without func.return")

    def run_module(self, module,  args: list[int], func_name: str = "my_func") -> int:
        """ Execute the same function multiple times with different argument sets.

        This method repeatedly calls `run_module`, once for each list of
        arguments in `batch_args`, effectively simulating batch execution.
        Each inner list represents the input arguments for one function call.

        Args:
            module:
                The xDSL ModuleOp containing the function to execute.

            batch_args:
                A list of argument lists. Each inner list contains the integer
                values that will be passed as inputs to one invocation of the
                function.

            func_name:
                The name of the function to execute. Defaults to "my_func".

        Returns:            
            list[int]:
                A list containing the return value of each function invocation,
                in the same order as `batch_args`.
      """

        self.globals = {op.sym_name.data: op for op in module.ops if op.name == "memref.global"}

        # module.body.blocks[0].ops usually contains top-level ops
        for top_block in module.body.blocks:
            for op in list(top_block.ops):
                if isinstance(op, func.FuncOp):
                    # Try to read the symbol name
                    sym = getattr(op, "sym_name", None)
                    name = None
                    if sym is not None:
                        name = _int_from_attr(getattr(sym, "data", None))  # probably not int, ignore
                        # better: sym.data is usually a string
                        if hasattr(sym, "data"):
                            name = sym.data
                        else:
                            name = str(sym)

                    if name == func_name or (name is None and func_name == "my_func"):
                        # assume single-block body for now
                        body_block = list(op.body.blocks)[0]

                        if len(args) != len(body_block.args):
                            raise RuntimeError(
                                f"Expected {len(body_block.args)} args, got {len(args)}"
                            )
                        for block_arg, value in zip(body_block.args, args):
                            self._set(block_arg, value)
                        return self.run_block(body_block)

        raise RuntimeError(f"Function not found: {func_name}")

    def run_module_batch(
        self,
        module,
        batch_args: list[list[int]],
        func_name: str = "my_func"
    ) -> list[int]:
        """Calls `run_module` once per argument list in `batch_args`, with a
        fresh environment each time, and returns the results in order."""
        results = []

        for args in batch_args:
            # reset environment for each independent execution
            self.env = {}
            result = self.run_module(module, args=args, func_name=func_name)
            results.append(result)

        return results

    def _eval_scf_if_region_yield(self, region) -> int:
        """Returns the value yielded by `region`'s single-operand `scf.yield`
        without executing the region's other ops (see module docstring)."""
        block = list(region.blocks)[0]
        last = list(block.ops)[-1]
        if last.name != "scf.yield" or len(last.operands) != 1:
            raise RuntimeError(f"Expected region to end with scf.yield(1), got: {last}")
        return self._get(last.operands[0])

