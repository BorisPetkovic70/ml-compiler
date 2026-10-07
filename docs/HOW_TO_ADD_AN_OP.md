# How to add a new `hc.*` operation

A mechanical checklist for extending the dialect: dialect → loader → lowering → interpreter →
tests. Every existing op followed this path; `HCMatmul` is used as the running example below
since it is the most invariant-dense op currently in the dialect. For *why* the pipeline is
shaped this way, see [DESIGN.md](DESIGN.md).

## 1. Dialect definition (`hc_dialect.py`)

Define the op as an `IRDLOperation` subclass with `@irdl_op_definition`, its operands/results
via `operand_def(...)`/`result_def(...)`, and a hand-written `verify_(self)` for any
cross-operand invariant the type system alone can't express (`TensorInt` doesn't bind
shapes across operands, so every op with a shape relationship between its operands needs its own
verifier — see DESIGN.md Section 2). An element-wise op needs neither a verifier of its own nor
separate scalar and tensor classes: give it `ScalarOrTensorInt` operands and result, and call
`_verify_elementwise(self)` from `verify_`, as `HCAdd` does.

```python
@irdl_op_definition
class HCMatmul(IRDLOperation):
    """(MxK) @ (KxN) -> (MxN) on the last two dims. Any leading batch dims
    are the same on lhs, rhs and res: (...xMxK) @ (...xKxN) -> (...xMxN)."""
    name = "hc.matmul"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        ...  # rank >= 2, same rank, element-type agreement, equal batch dims, MxK · KxN -> MxN
```

**Register it in the `HiCompiler` tuple** at the bottom of the file — an op that isn't
registered here isn't part of the dialect at all, regardless of how correctly it's defined
above:

```python
HiCompiler = Dialect(
    "hc",
    (
        HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,   # element-wise ops
        HCMatmul,
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
    a_dims, b_dims = shape_of(a.type), shape_of(b.type)
    if len(a_dims) < 2 or len(b_dims) < 2:
        raise NotImplementedError(...)  # names the node
    batch, m = a_dims[:-2], a_dims[-2]
    n = b_dims[-1]
    res_ty = builtin.TensorType(i32, batch + [m, n])
    hc = HCMatmul(operands=[a, b], result_types=[res_ty])
    emit(hc)
    continue
```

`emit` verifies the op and binds its result to the node's output. A binary element-wise op
needs no branch: add a row to `_BINARY_OPS`. Its result type comes from `broadcast_type(a, b)`,
which is `i32` for two scalars and a tensor otherwise.

## 3. Lowering (`middle_end/hc_lowering.py`)

`LowerHCPattern.match_and_rewrite` lowers `hc.matmul` with its own loop nest and every other
`hc` op through the `_KERNELS` table. An `hc` op with neither raises `NotImplementedError`.

**An element-wise op is a kernel plus a table row.** A kernel takes the op's operands as scalar
values and returns `(ops, result_value)` for one element:

```python
def _add(a, b):
    add = arith.AddiOp(a, b)
    return [add], add.result

_KERNELS = {
    "hc.add": _add,
    ...
}
```

That is the whole lowering. When the op's result is a scalar, the kernel's ops replace it. When
the result is a tensor, `_build_tensor_elementwise_nest` builds one loop per dim with
`_build_nest`, reads each operand with broadcasting, and runs the kernel in the innermost body.
A kernel may hold control flow of its own: `hc.max` is an `scf.if` and `hc.pow` an `scf.for`.

**Any other op** needs its own nest. Write a module-level helper that returns
`(new_ops, result_value)`, as `_build_matmul_nest` does, and add a branch for the op's name
next to `hc.matmul`'s. The shared `rewriter.replace(op, new_ops=..., new_results=[...],
safe_erase=True)` at the end of the method then swaps the op for the new ones.

## 4. Interpreter (`simulator/interpreter.py`)

Add a branch in `run_block` matching the op's name (for evaluating it directly, before
lowering) — and, if the lowering introduces operation kinds the interpreter doesn't already
handle, add support for those too. `hc.matmul`'s addition, for example, required both a native
`hc.matmul` branch (the reference `_matmul` helper, deliberately *not*
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
  `hc.` op survives. The `hc.relu` test:

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


