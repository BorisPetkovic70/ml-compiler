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
    """(MxK) * (KxN) -> (MxN), or batched (BxMxK) * (BxKxN) -> (BxMxN)"""
    name = "hc.matmul"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        ...  # rank-2-or-3 check, same-rank check, element-type agreement, MxK · KxN -> MxN
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
        HCMatmul, HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor,  # tensor ops
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
    a_dims = [_dim_as_int(d) for d in a.type.shape]
    b_dims = [_dim_as_int(d) for d in b.type.shape]
    batch, m = a_dims[:-2], a_dims[-2]
    n = b_dims[-1]
    res_ty = builtin.TensorType(i32, batch + [m, n])
    hc = HCMatmul(operands=[a, b], result_types=[res_ty])
    ops.append(hc)
    env[node.output[0]] = hc.results[0]
    continue
```

If the op has scalar/vector/tensor variants (like `Add`/`Mul`), dispatch on `_is_tensor(...)` /
`_is_vec(...)` for each operand, following the existing `Add`/`Sub`/`Mul` branches.

## 3. Lowering (`middle_end/hc_lowering.py`)

**This is the step most likely to fail silently if skipped.** `LowerHCPattern.match_and_rewrite`
opens with a dispatch guard:

```python
if op.name not in (
    "hc.add", "hc.mul", "hc.sub", "hc.relu", "hc.pow", "hc.max", "hc.min",
    "hc.add_vec", "hc.sub_vec", "hc.mul_vec", "hc.mul_vec_vec", "hc.relu_vec",
    "hc.matmul", "hc.add_tensor", "hc.sub_tensor", "hc.mul_tensor", "hc.relu_tensor",
):
    return
```

**An op name left out of this tuple is not an error — the pattern simply returns without
touching it, and the unlowered `hc.*` op survives silently into whatever comes next in the
pipeline**, surfacing as a confusing failure far from the actual cause (usually inside
bufferization or the backend, which don't recognize `hc.*` ops at all). A name that *is* in the
tuple but has no branch behaves the same way — the method falls off the end and returns — so
adding the name first is not a guard by itself. The guard is `check_lowering()` in
`test_lowering.py`, which checks that no `hc.` op survives lowering; give the new op a test
there (step 5).

Then add the branch itself, building the replacement operation(s) and calling
`rewriter.replace(op, new_ops=[...], new_results=[...], safe_erase=True)` (not the deprecated
`replace_op`). If the lowering is large enough that inlining it would clutter the dispatch
method (`hc.matmul`'s nest is ~40 lines), factor it into a module-level helper function
(`_build_matmul_nest`) and call that from the branch. An element-wise tensor op needs no loop
code of its own: pass a `compute(a, b)` function for one element to
`_build_tensor_elementwise_nest`, which builds one loop per dim with `_build_nest`.

## 4. Interpreter (`simulator/interpreter.py`)

Add a branch in `run_block` matching the op's name (for evaluating it directly, before
lowering) — and, if the lowering introduces operation kinds the interpreter doesn't already
handle, add support for those too. `hc.matmul`'s addition, for example, required both a native
`hc.matmul` branch (the reference `_matmul`/`_batched_matmul` helpers, deliberately *not*
derived from the lowering, so the two can't share a bug) and tensor support in general (`tensor.extract`/`tensor.insert`,
bounds-checked).

## 5. Tests

A new op gets three tests, each in an existing file. If the op has the same shape as an
existing one, each test is just a new row in a parametrized table.

- **`tests/test_dialect.py`:** a positive case that builds and verifies the op, and a negative
  case (`pytest.raises`) that its `verify_()` rejects.

- **`tests/test_lowering.py`:** one FileCheck test. Write the op in generic form, run it
  through `check_lowering`, and match the lowered IR with `// CHECK:` lines. Captures such as
  `%[[ZERO:.*]]` show how the new ops are wired together. `check_lowering` also checks that no
  `hc.` op survives (step 3's guard). The `hc.relu` test:

  ```python
  def test_relu():
      check_lowering("""
      func.func @f(%x: i32) -> i32 {
        %r = "hc.relu"(%x) : (i32) -> i32
        func.return %r : i32
      }""", """
      // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
      // CHECK: %[[R:.*]] = arith.maxsi %x, %[[ZERO]] : i32
      // CHECK: func.return %[[R]] : i32
      """)
  ```

- **`tests/test_oracle.py`:** one row, for example `("hc.relu", "i32", -3, 0)`, or a short
  `check_oracle` test. The interpreter runs the op before and after the whole middle end, and
  both results must equal the expected value. A CHECK test shows what the lowering emits; this
  shows that it computes the right thing. Where NumPy has a reference (as for `hc.matmul`),
  compare against it rather than a hand-written implementation that could share a bug.

## 6. Verify

Run the full suite and CI's coverage gate:

```bash
../.venv/bin/python -m pytest -q
../.venv/bin/python -m pytest --cov=hc_dialect --cov=middle_end --cov=back_end \
  --cov=simulator --cov-report=term-missing --cov-fail-under=85 -q
```


