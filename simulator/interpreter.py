from typing import Any
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.dialects import func
from xdsl.dialects.builtin import VectorType

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
    """
    arith.constant dense<...> : vector<Nxi32> -> list[int], else None.
    """
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
    """Apply f element-wise, broadcasting a scalar against a vector (list)."""
    if isinstance(a, list) and isinstance(b, list):
        return [f(x, y) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [f(x, b) for x in a]
    if isinstance(b, list):
        return [f(a, y) for y in b]
    return f(a, b)


def _vec_len(vec_type: VectorType) -> int:
    n = 1
    for d in vec_type.shape:
        n *= int(getattr(d, "data", d))
    return n

# -----------------------------
# Interpreter
# -----------------------------

class Interpreter:
    def __init__(self):
        # SSAValue is not always hashable across versions -> use id()
        self.env: dict[int, object] = {}

    def _get(self, v: SSAValue):
        return self.env[id(v)]

    def _set(self, v: SSAValue, value) -> None:
        # print(f"Set SSAValue: {v} = {value}")
        self.env[id(v)] = value

    def run_block(self, block) -> int:
        """
        Execute a single basic block (straight-line).
        Returns the integer from func.return.
        """
        for op in list(block.ops):
            name = op.name

            # --- constants ---
            if name == "arith.constant":
                dense = _dense_values(op)
                if dense is not None:
                    self._set(op.results[0], dense)
                    continue
                val = _const_value(op)
                if val is None:
                    raise RuntimeError(f"Couldn't read constant value from op: {op}")
                self._set(op.results[0], val)
                continue

            # --- high-level HC ops (scalar and vector variants share semantics) ---
            if name in ("hc.add", "hc.add_vec"):
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), lambda p, q: p + q))
                continue

            if name in ("hc.sub", "hc.sub_vec"):
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), lambda p, q: p - q))
                continue

            if name in ("hc.mul", "hc.mul_vec", "hc.mul_vec_vec"):
                a, b = op.operands
                self._set(op.results[0], _elt_binop(self._get(a), self._get(b), lambda p, q: p * q))
                continue

            if name in ("hc.relu", "hc.relu_vec"):
                (x,) = op.operands
                self._set(op.results[0], _elt_binop(self._get(x), 0, max))
                continue

            if name == "hc.pow":
                a, b = op.operands
                self._set(op.results[0], self._get(a) ** self._get(b))
                continue

            if name == "hc.max":
                a, b = op.operands
                self._set(op.results[0], max(self._get(a), self._get(b)))
                continue

            if name == "hc.min":
                a, b = op.operands
                self._set(op.results[0], min(self._get(a), self._get(b)))
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

                # In xDSL: sgt == 4; lgt == 2
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
                # Try: recognize "pow lowering" pattern and shortcut it.
                if self._try_eval_pow_lowering(op):
                    continue
                raise RuntimeError("Unsupported scf.for (not recognized as pow lowering)")
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
        """ Execute a single function from the given module using the interpreter.

        This method locates `func.func @func_name` inside the module, binds the
        provided runtime arguments to the function's entry block arguments,
        and interprets the block sequentially until a `func.return` operation
        is encountered.

        Args:
            module:
                The xDSL ModuleOp containing the function to execute.

            args:
                A list of integer values that will be bound to the function's
                input arguments (entry block arguments). The number of elements
                must match the number of function parameters.

            func_name:
                The name of the function to execute. Defaults to "my_func".

        Returns:
            int:
                The integer value returned by the `func.return` operation.
        """
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
        results = []

        for args in batch_args:
            # reset environment for each independent execution
            self.env = {}
            result = self.run_module(module, args=args, func_name=func_name)
            results.append(result)

        return results

    def _try_eval_pow_lowering(self, op) -> bool:
        """
        Recognize and evaluate the specific lowering:
          %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (i32) {
            %m = arith.muli %acc, %base : i32
            scf.yield %m : i32
          }
        If matched: set scf.for result(s) and return True, else False.
        """
        # Must have exactly 1 iter_arg result
        if len(op.results) != 1:
            return False

        # Must have lb, ub, step at least
        if len(op.operands) < 4:
            return False

        lb = self._get(op.operands[0])
        ub = self._get(op.operands[1])
        step = self._get(op.operands[2])
        init = self._get(op.operands[3])  # first iter_arg init

        # Typical pow lowering: lb=0, step=1, ub>=0
        if lb != 0 or step != 1 or ub < 0:
            return False

        # Body: single block
        try:
            body_block = op.regions[0].blocks[0]
        except Exception:
            return False

        ops = list(body_block.ops)
        # Expect exactly: [arith.muli, scf.yield]
        if len(ops) != 2:
            return False

        mul, yld = ops
        if mul.name != "arith.muli" or yld.name != "scf.yield":
            return False

        # scf.yield must yield the mul result
        if len(yld.operands) != 1 or yld.operands[0] is not mul.results[0]:
            return False

        # mul operands should be: (iter_arg_block_arg, base_value)
        # We don't want to fully interpret block args; just identify the "base" operand:
        if len(mul.operands) != 2:
            return False

        a, b = mul.operands

        # In xDSL, block args are typically instances of BlockArgument.
        if isinstance(a, BlockArgument) and not isinstance(b, BlockArgument):
            base_val = self._get(b)
        elif isinstance(b, BlockArgument) and not isinstance(a, BlockArgument):
            base_val = self._get(a)
        else:
            # Either both are block args or both are not -> not the simple pow pattern
            return False

        # Evaluate
        result = init * (base_val ** ub)

        self._set(op.results[0], result)
        return True

    def _eval_scf_if_region_yield(self, region) -> int:
        """
        Simplified: Assume region has 1 block and the last op is scf.yield with 1 operand.
        We ignore everything else in the region and just read the yielded SSA value.
        """
        block = list(region.blocks)[0]
        last = list(block.ops)[-1]
        if last.name != "scf.yield" or len(last.operands) != 1:
            raise RuntimeError(f"Expected region to end with scf.yield(1), got: {last}")
        return self._get(last.operands[0])

