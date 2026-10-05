"""Read-only analyses.

- `is_last_use` answers whether a value is dead once an op has run, across
  nested `scf` regions. Bufferization uses it to decide whether a write may
  happen in place.
- The debug reports print a use-def listing and block-local liveness sets for
  each function's top-level blocks. They run only when
  `MiddleEndPipelineConfig.run_analysis` is set, and no pass consumes their
  output."""
from typing import Any, Iterable
from xdsl.dialects import func, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Operation, SSAValue

# -----------------------------------------------------------------------------
# Last-use query (region-aware)
# -----------------------------------------------------------------------------
def _count_uses(op: Operation, value: SSAValue) -> int:
    """How many times `op`, or any op nested inside it, uses `value`."""
    return sum(o is value for nested in op.walk() for o in nested.operands)


def is_last_use(value: SSAValue, op: Operation) -> bool:
    """True if `op` uses `value` exactly once and nothing can read `value`
    after `op` has run.

    Walks outwards from `op` to the block that defines `value`. `value` is
    still needed if:
    - `op` uses it a second time, directly or inside its own regions;
    - an op later in any block on the way out uses it, directly or nested;
    - the walk leaves an `scf.for` body, because the next iteration runs
      `op` again and expects `value` unchanged.
    """
    if _count_uses(op, value) != 1:
        return False
    def_block = value.owner if isinstance(value.owner, Block) else value.owner.parent
    node = op
    while True:
        later = node.next_op
        while later is not None:
            if _count_uses(later, value):
                return False
            later = later.next_op
        if node.parent is def_block:
            return True
        node = node.parent_op()
        if isinstance(node, scf.ForOp):
            return False

# -----------------------------------------------------------------------------
# Analysis 1: Use-def report
# -----------------------------------------------------------------------------
def print_use_def(block: Block) -> None:
    print("\n--- Use-def report (per SSA result) ---")
    for op in block.ops:
        for res in _collect_results(op):
            users: list[str] = []
            for use in _iter_uses(res):
                uop = _use_owner_op(use)
                if uop is None:
                    users.append("<unknown-user>")
                else:
                    users.append(_fmt_op(uop))
            users_s = ", ".join(users) if users else "(no uses)"
            print(f"{_fmt_val(res)} defined by {_fmt_op(op)}  -> used by: {users_s}")

# -----------------------------------------------------------------------------
# Analysis 2: Block-local liveness (straight-line)
# -----------------------------------------------------------------------------
def compute_liveness(block: Block):
    """Returns `live_after`, where `live_after[i]` maps id(value) -> value for
    every SSA value live after `block.ops[i]`."""
    ops = list(block.ops)
    live_after: list[dict[int, SSAValue]] = [dict() for _ in ops]

    live: dict[int, SSAValue] = {}

    # Seed with values used by the terminator (usually func.return)
    if ops:
        term = ops[-1]
        for v in _collect_operands(term):
            live[_val_key(v)] = v

    # Walk backwards
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]

        # live_after for this op = current live (after processing later ops)
        live_after[i] = dict(live)

        # Remove defs (results) from live (killing)
        for res in _collect_results(op):
            live.pop(_val_key(res), None)

        # Add uses (operands) to live
        for v in _collect_operands(op):
            live[_val_key(v)] = v

    return live_after


def print_liveness(block: Block) -> None:
    live_after = compute_liveness(block)

    print("\n--- Block liveness (live set AFTER each op) ---")
    ops = list(block.ops)
    for i, op in enumerate(ops):
        live_vals = sorted((_fmt_val(v) for v in live_after[i].values()))
        live_s = ", ".join(live_vals) if live_vals else "(empty)"
        print(f"[{i:02d}] after {_fmt_op(op):>12} : {live_s}")

def analyze(module: ModuleOp):
    # Analyze each function body block
    for top_block in module.body.blocks:
        for op in top_block.ops:
            if isinstance(op, func.FuncOp):
                print(f"\n============================")
                print(f"Analyzing Function: {op.sym_name.data if hasattr(op, 'sym_name') else 'func'}")
                print(f"============================")

                for body_block in op.body.blocks:
                    print_use_def(body_block)
                    print_liveness(body_block)

# Helper functions
def _iter_uses(val: SSAValue) -> Iterable[Any]:
    uses = getattr(val, "uses", None)
    if uses is None:
        return []
    return uses

def _use_owner_op(use: Any) -> Operation | None:
    """Returns the operation that owns `use`, or None."""
    for attr in ("operation", "op", "owner"):
        u = getattr(use, attr, None)
        if isinstance(u, Operation):
            return u
    # some Use objects store the operation under .operation on a field
    return None

def _fmt_val(v: SSAValue) -> str:
    try:
        return str(v)  # often prints as %0
    except Exception:
        return f"<ssa@{id(v):x}>"

def _fmt_op(op: Operation) -> str:
    # compact one-liner label
    return f"{op.name}"

def _val_key(v: SSAValue) -> int:
    """Returns id(v), a dictionary key that works whether or not SSAValue is
    hashable."""
    return id(v)

def _collect_operands(op: Operation) -> list[SSAValue]:
    return list(getattr(op, "operands", []))

def _collect_results(op: Operation) -> list[SSAValue]:
    return list(getattr(op, "results", []))
