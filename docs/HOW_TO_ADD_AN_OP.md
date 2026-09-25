# How to add a new `hc.*` operation

A mechanical checklist for extending the dialect: dialect → loader → lowering → interpreter →
tests. Every existing op followed this path; `HCMatmul` is used as the running example below
since it is the most invariant-dense op currently in the dialect. For *why* the pipeline is
shaped this way, see [DESIGN.md](DESIGN.md).

## 1. Dialect definition (`hc_dialect.py`)

Define the op as an `IRDLOperation` subclass with `@irdl_op_definition`, its operands/results
via `operand_def(...)`/`result_def(...)`, and a hand-written `verify_(self)` for any
cross-operand invariant the type system alone can't express (`VecInt`/`TensorInt` don't bind
shapes across operands, so every op with a shape relationship between its operands needs its own
verifier — see DESIGN.md Section 2):

```python
@irdl_op_definition
class HCMatmul(IRDLOperation):
    """2D matrix multiply: (MxK) * (KxN) -> (MxN)"""
    name = "hc.matmul"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        ...  # rank-2 check, element-type agreement, MxK · KxN -> MxN
```

**Register it in the `HiCompiler` tuple** at the bottom of the file — an op that isn't
registered here isn't part of the dialect at all, regardless of how correctly it's defined
above:

```python
HiCompiler = Dialect(
    "hc",
    (
        HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,      # scalar ops
        HCAddVec, HCSubVec, HCMulVec, HCReluVec, HCMulVecVec,  # vector ops
        HCMatmul,                                              # tensor ops
    ),
    (),  # attrs
)
```

## 2. Loader dispatch (`front_end/loader.py`)

Add a branch in `import_onnx_to_hc_module`'s node loop that recognizes the relevant ONNX
`op_type`, reads the already-bound operand values via `get(...)`, and builds the new op:

```python
if node.op_type == "MatMul":
    a = get(node.input[0])
    b = get(node.input[1])
    m = _dim_as_int(list(a.type.shape)[0])
    n = _dim_as_int(list(b.type.shape)[1])
    res_ty = builtin.TensorType(i32, [m, n])
    hc = HCMatmul(operands=[a, b], result_types=[res_ty])
    ops.append(hc)
    env[node.output[0]] = hc.results[0]
    continue
```

If the op needs a scalar/vector variant split (like `Add`/`Mul`), dispatch on `_is_vec(...)` for
each operand, following the existing `Add`/`Sub`/`Mul` branches as the pattern.

## 3. Lowering (`middle_end/hc_lowering.py`)

**This is the step most likely to fail silently if skipped.** `LowerHCPattern.match_and_rewrite`
opens with a dispatch guard:

```python
if op.name not in (
    "hc.add", "hc.mul", "hc.sub", "hc.relu", "hc.pow", "hc.max", "hc.min",
    "hc.add_vec", "hc.sub_vec", "hc.mul_vec", "hc.mul_vec_vec", "hc.relu_vec",
    "hc.matmul",
):
    return
```

**An op name left out of this tuple is not an error — the pattern simply returns without
touching it, and the unlowered `hc.*` op survives silently into whatever comes next in the
pipeline**, surfacing as a confusing failure far from the actual cause (usually inside
bufferization or the backend, which don't recognize `hc.*` ops at all). Add the new op's name to
this tuple *before* writing its lowering branch, so a missing branch fails immediately as "no
matching branch" rather than passing through unnoticed.

Then add the branch itself, building the replacement operation(s) and calling
`rewriter.replace(op, new_ops=[...], new_results=[...], safe_erase=True)` (not the deprecated
`replace_op`). If the lowering is large enough that inlining it would clutter the dispatch
method (`hc.matmul`'s nest is ~40 lines), factor it into a module-level helper function
(`_build_matmul_nest`) and call that from the branch.

## 4. Interpreter (`simulator/interpreter.py`)

Add a branch in `run_block` matching the op's name (for evaluating it directly, before
lowering) — and, if the lowering introduces operation kinds the interpreter doesn't already
handle, add support for those too. `hc.matmul`'s addition, for example, required both a native
`hc.matmul` branch (an independent triple loop, deliberately *not* derived from the lowering, so
the two can't share a bug) and tensor support in general (`tensor.extract`/`tensor.insert`,
bounds-checked).

## 5. Tests

Follow the existing three-file, three-level pattern — add one row to each relevant
parametrized table rather than writing a whole new test function; this is the intended
low-friction path for anything that fits the existing shape:

- **`tests/test_dialect.py`** — construction plus `verify_()`. If the op has a `verify_()`
  (most do), add both a positive case and a negative (`pytest.raises`) case *for every distinct
  raise branch*. After adding the tests, run:

  ```bash
  ../.venv/bin/python -m pytest --cov=hc_dialect --cov-report=term-missing -q
  ```

  and check for any uncovered `raise` line in the new `verify_()`. This repository holds
  `hc_dialect.py` to 100% line coverage; a new op should not regress that.

- **`tests/test_lowering.py`** — structural: assert the new op's lowering produces the expected
  operation kinds, and that no `hc.*` op survives. Where a bug could hide behind "the right kind
  of operation is present but wired up wrong" (a common failure mode — see `hc.matmul`'s own
  wiring tests, which check load/store indices by identity against the actual loop induction
  variables, not just that `tensor.extract` appears somewhere), assert on the actual operation's
  attributes/operands via `find_op`/`find_ops`, not just membership in `entry_op_names`.

- **`tests/test_interpreter.py`** — semantic: run the same IR through the interpreter *before*
  and *after* lowering and assert identical results. This is what actually proves the lowering
  correct, not merely plausible — a structural check can confirm the right operations appear
  without confirming they're wired together correctly. If a NumPy-comparable reference exists
  (as for `hc.matmul`), gate against that instead of a second hand-written implementation, to
  avoid both sides sharing the same bug.

## 6. Verify, then hand off

Run the full suite and the coverage check one more time:

```bash
../.venv/bin/python -m pytest -q
../.venv/bin/python -m pytest --cov=hc_dialect --cov-report=term-missing -q
```


