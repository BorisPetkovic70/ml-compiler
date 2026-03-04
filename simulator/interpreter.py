from typing import Any
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.dialects import func

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
# -----------------------------
# Interpreter
# -----------------------------

class Interpreter:
    def __init__(self):
        # SSAValue is not always hashable across versions -> use id()
        self.env: dict[int, int] = {}

    def _get(self, v: SSAValue) -> int:
        return self.env[id(v)]

    def _set(self, v: SSAValue, value: int) -> None:
        print(f"Set SSAValue: {v} = {int(value)}")
        self.env[id(v)] = int(value)

    def run_block(self, block) -> int:
        """
        Execute a single basic block (straight-line).
        Returns the integer from func.return.
        """
        for op in list(block.ops):
            name = op.name

            # --- constants ---
            if name == "arith.constant":
                val = _const_value(op)
                if val is None:
                    raise RuntimeError(f"Couldn't read constant value from op: {op}")
                self._set(op.results[0], val)
                continue

            # --- high-level HC ops ---
            if name == "hc.add":
                a, b = op.operands
                self._set(op.results[0], self._get(a) + self._get(b))
                continue

            if name == "hc.sub":
                a, b = op.operands
                self._set(op.results[0], self._get(a) - self._get(b))
                continue

            if name == "hc.mul":
                a, b = op.operands
                self._set(op.results[0], self._get(a) * self._get(b))
                continue

            if name == "hc.relu":
                (x,) = op.operands
                self._set(op.results[0], max(self._get(x), 0))
                continue

            if name == "hc.pow":
                a, b = op.operands
                self._set(op.results[0], self._get(a) ** self._get(b))
                continue

            # --- lowered arith ops (so you can run after lowering too) ---
            if name == "arith.addi":
                a, b = op.operands
                self._set(op.results[0], self._get(a) + self._get(b))
                continue

            if name == "arith.subi":
                a, b = op.operands
                self._set(op.results[0], self._get(a) - self._get(b))
                continue

            if name == "arith.muli":
                a, b = op.operands
                self._set(op.results[0], self._get(a) * self._get(b))
                continue

            if name == "arith.maxsi":
                a, b = op.operands
                self._set(op.results[0], max(self._get(a), self._get(b)))
                continue

            if name == "scf.for":
                # Try: recognize "pow lowering" pattern and shortcut it.
                if self._try_eval_pow_lowering(op):
                    continue
                raise RuntimeError("Unsupported scf.for (not recognized as pow lowering)")

            # --- return ---
            if name == "func.return":
                if len(op.operands) != 1:
                    raise RuntimeError(f"Expected func.return with 1 operand, got: {op}")
                return self._get(op.operands[0])

            raise RuntimeError(f"Unsupported op in interpreter: {name}\nOp: {op}")

        raise RuntimeError("Block ended without func.return")

    def run_module(self, module, func_name: str = "my_func") -> int:
        """
        Find func.func @func_name and execute its first block.
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
                        return self.run_block(body_block)

        raise RuntimeError(f"Function not found: {func_name}")

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


