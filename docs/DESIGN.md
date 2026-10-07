# Design

How the compiler is structured, and why. For setup and commands see [README.md](../README.md);
for extending the dialect see [HOW_TO_ADD_AN_OP.md](HOW_TO_ADD_AN_OP.md).

## 1. Layers

```
ONNX graph
  │  front_end/loader.py        which hc op is each ONNX node?
  ▼
hc.* dialect                    WHAT is computed (one op per ONNX node)
  │  middle_end/hc_lowering.py  HOW: loops over values (value semantics)
  ▼
arith / scf / tensor
  │  middle_end/bufferization.py  WHERE: tensors become buffers (memory semantics)
  ▼
arith / scf / memref
  │  back_end/back_end.sh       mlir-opt → mlir-translate → llc → clang
  ▼
native executable  (+ back_end/harness_gen.py's generated C main)
```

`hc` exists so that the question "which operation is this?" (the loader's job) is separate from
"what loops and memory implement it?" (lowering's job). It is ONNX-shaped but already xDSL IR,
so it can be verified, printed and interpreted with the same tools as everything below it.

Each stage boundary is also a test boundary: the interpreter (§6) runs the IR on both sides of
a transform, and the results must match.

## 2. Scalars and tensors

These are two separate kinds of value, each with its own op family (`hc.add`,
`hc.add_tensor`, …), because each maps onto hardware differently. Every ONNX value of rank 1
or more is a tensor:

| Kind | Type | Hardware | Lowering |
|---|---|---|---|
| scalar | `i32` | general register | one `arith` op (`hc.add` → `arith.addi`) |
| tensor | `tensor<Nxi32>`, `tensor<MxNxi32>`, … | none | an `scf.for` loop nest |

The type constraint `TensorInt` only fixes the element type. It does not relate the
shapes of different operands, so every op with a shape rule (same shape; a broadcast shape;
`MxK · KxN → MxN`; a matching batch dim) enforces it in a hand-written `verify_()`. Keeping
families separate keeps each verifier small.

The tensor ops are the first whose lowering adds structure the source op didn't have:

- elementwise (`add/sub/mul/relu_tensor`): one loop per dim, for any rank >= 1; a map with no
  reduction. `_build_nest` builds the loops and the op supplies only the body.
- `hc.matmul`: `i, j, k` loops, where `k` is the reduction. Rank 3 adds one outer batch loop,
  whose induction variable is prepended to every index.

**Broadcasting.** `hc.add_tensor`, `hc.sub_tensor` and `hc.mul_tensor` follow NumPy's rules:
the operand shapes are aligned at the last dim, and each pair of dims must be equal or one of
them 1. An operand may also be a scalar `i32`. `broadcast_shape(a, b)` in `hc_dialect.py`
computes the result shape; the verifier and the loader both use it.

The lowering loops over the result's shape and reads each operand as far as its own shape
reaches (`_broadcast_extract`). A tensor operand is indexed by the trailing induction
variables, with a constant `0` where its dim is 1 and the result's is not. A scalar operand is
used directly. Nothing is copied or expanded:

```mlir
// %a: tensor<2x3xi32>, %b: tensor<2x1xi32>, inside the i/j nest
%x = tensor.extract %a[%i, %j] : tensor<2x3xi32>
%y = tensor.extract %b[%i, %c0] : tensor<2x1xi32>
```

`hc.max`/`hc.min` lower to `arith.cmpi` + `scf.if`, not `arith.maxsi`/`minsi`. That follows the
tutorial in `doc_upload/`, which uses them to introduce structured control flow. The
interpreter's `scf.if` shortcut (§7) relies on this exact shape.

## 3. Value semantics vs memory semantics

This is the central idea of the middle end.

- **Value semantics (tensors).** An SSA value is defined once and never changes.
  `tensor.insert %v into %t[i, j]` does not modify `%t`; it returns a *new* tensor. So a loop
  that builds a tensor must pass the latest version from one iteration to the next. `scf.for`
  does this with **`iter_args`**: values the loop body receives, then hands back via
  `scf.yield`.
- **Memory semantics (memrefs).** A memref is a buffer. `memref.store` writes into it in place;
  every value referring to that buffer sees the write.

Lowered `hc.matmul` (value semantics), simplified:

```mlir
%empty = tensor.empty() : tensor<4x4xi32>
%C = scf.for %i = %c0 to %c4 step %c1 iter_args(%Ci = %empty) -> (tensor<4x4xi32>) {
  %Cj = scf.for %j = %c0 to %c4 step %c1 iter_args(%Cij = %Ci) -> (tensor<4x4xi32>) {
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
      %a = tensor.extract %A[%i, %k] : tensor<4x4xi32>
      %b = tensor.extract %B[%k, %j] : tensor<4x4xi32>
      %p = arith.muli %a, %b : i32
      %s = arith.addi %acc, %p : i32
      scf.yield %s : i32
    }
    %next = tensor.insert %sum into %Cij[%i, %j]
    scf.yield %next : tensor<4x4xi32>
  }
  scf.yield %Cj : tensor<4x4xi32>
}
```

The result starts as `tensor.empty`: a tensor with a shape but unspecified contents. The nest
writes every element and reads none, so no initial value is needed.

After bufferization (memory semantics):

```mlir
%C = memref.alloc() : memref<4x4xi32>
scf.for %i = %c0 to %c4 step %c1 {
  scf.for %j = %c0 to %c4 step %c1 {
    %sum = scf.for %k ... iter_args(%acc = %c0_i32) -> (i32) { ... memref.load ... }
    memref.store %sum, %C[%i, %j]
  }
}
```

The `i`/`j` loops no longer carry anything: every store hits the same buffer. The `k` loop still
carries its `i32` accumulator, because that is a register-level reduction, not memory.

Replacing a copying insert with an in-place store is only correct if nobody still needs the old
tensor value. §4 covers how the pass ensures this.

## 4. Bufferization (hand-written)

MLIR provides a general pass for this (`one-shot-bufferize`). This project writes its own
(`middle_end/bufferization.py`) for two reasons:

1. **The memref IR has to exist in Python.** The interpreter must execute the bufferized IR
   to check it against the pre-bufferization result. `harness_gen.py` must see the bufferized
   entry signature (memref arguments and results) to generate a matching C harness. Both
   happen before `mlir-opt` ever runs, so bufferizing inside `mlir-opt` would be too late.
2. **Learning goal.** Writing the tensor→memref conversion by hand is part of the point of the
   project.

**What it does**, per function that touches tensors. The body is rebuilt (clone and translate)
rather than edited in place, because an `scf.for`'s number of `iter_args` can't be changed in place:

| Before | After |
|---|---|
| tensor function argument | memref argument (read-only; not owned) |
| `tensor.empty` (an op's result) | `memref.alloc`, left unfilled |
| tensor constant (a weight) | module-level `memref.global constant` + `memref.get_global` (read-only; not owned) |
| `tensor.extract %t[idx]` | `memref.load %m[idx]` |
| `tensor.insert %v into %t[idx]` | `memref.store %v, %m[idx]`, where `%m` is `%t`'s buffer or a copy of it (see below) |
| tensor `iter_args` / yields | dropped (scalar ones kept); the loop writes into its init's buffer or a copy of it |
| returned tensor | returned memref; the **caller owns and frees** it. A buffer the function doesn't own (an argument or a weight) is first copied into a new one (`memref.alloc` + `memref.copy`) |
| other allocated buffers | `memref.dealloc` just before `func.return` |

So a buffer is one of three kinds: an input (the caller's), a temporary this pass allocated
(owned), or a weight in the executable's read-only data. Only owned buffers are written to or
deallocated.

**In place or copy.** An insert, or a loop carrying a tensor, overwrites a buffer. That is done
in the tensor's own buffer only when both hold:

- the function owns the buffer. Writing to an argument would modify the caller's input, and
  writing to a weight would write to read-only data;
- the tensor is dead after the write, so no later use can see it. `analysis.is_last_use`
  decides this: the value must not be used by a later op, by the loop's own body (for an
  init), or from inside an `scf.for` body that doesn't define it, where the next iteration
  needs it unchanged. A read *before* the write is fine.

Otherwise the pass emits `memref.alloc` + `memref.copy` and writes into the copy, which the
function owns like any other temporary.

**It refuses (`NotImplementedError`) instead of guessing when:**

- a write inside a loop body needs a copy. Copies are placed in the function's top-level
  block only: every owned buffer is freed there, just before `func.return`, and a buffer
  allocated once per iteration could not be;
- a loop yields a different buffer than the one it carries;
- the function body has more than one block;
- a `tensor.empty` sits inside a nested region;
- any other op touches a tensor (for example, an `hc.*` op that was never lowered).

Neither the copies nor the refusals are triggered by IR the current lowering produces,
including a chained `hc.matmul` → `hc.relu_tensor` and a matmul with a weight. The
`dense_layer` sample model (`relu(X @ W + B)`) shows all three buffer kinds at once: `X` is an
input, `W` and `B` are weights, and the matmul and add results are temporaries.

## 5. Pass ordering and verification

`MiddleEndPipeline.apply_passes` runs: **lowering → bufferization → constant folding → constant CSE → DCE**,
each switchable via `MiddleEndPipelineConfig`.

- Bufferization needs lowering first: the `tensor.*` ops only exist after it.
- Constant CSE (`middle_end/cse.py`) runs after folding, so it also merges the constants
  folding creates. Each lowering builds its own loop bounds and zeros; the pass keeps one
  `arith.constant` per value and type and moves it to the top of the function, where it is
  visible in every loop body.
- DCE runs last so it can remove constants that folding or bufferization left unused.
- `FoldArithInts` folds `addi`/`subi`/`muli`/`maxsi` on scalar/vector constants and a
  `vector.broadcast` of a constant. `apply_dce` repeats until nothing changes, removing
  unused `arith.*`/`vector.broadcast` ops from the function's top-level block.

The pipeline **never calls `module.verify()`**. Whether and when to verify is the caller's
choice: `tests/conftest.py::lower()` verifies right after the pipeline, while `hc_main.py`
does not verify and `hc_interpret.py` compares the interpreter's result before and after.

## 6. The ABI boundary

A calling convention (ABI) fixes how arguments and results travel between caller and callee.
The C harness must declare the compiled function with exactly the right signature, and that
signature varies per model. So `harness_gen.py` generates one harness per model, choosing
between two conventions:

- **"Register ABI"** (no memref in the signature): calls `my_func` directly with ordinary
  by-value C `int` arguments. Only an all-scalar model has such a signature.
- **C-interface ABI** (any memref): calls `_mlir_ciface_my_func`, a wrapper that
  `mlir-opt --llvm-request-c-wrappers` generates. Each memref is passed as a pointer to a
  *descriptor struct* `{allocated, aligned, offset, sizes[R], strides[R]}`. A memref result
  becomes a hidden first out-parameter, because the struct is too large to return by value.
  Scalars stay by value. The callee `malloc`s the result, so the harness frees it.

  ```c
  void _mlir_ciface_my_func(MemRef2D_i32 *result, MemRef2D_i32 *a0, MemRef2D_i32 *a1);
  ```

Because bufferization changes a tensor signature into a memref one, the harness is generated
*after* the middle end. The generator raises an error for any signature it doesn't model: a
type other than an integer or a memref, a memref of rank 0 or ≥4, a non-`i8/16/32/64` element
type, or a memref argument with a non-memref result.

Weights (`memref.global`) live in the executable's read-only data. clang links a
position-independent executable by default, so `back_end.sh` runs `llc` with
`-relocation-model=pic`; without it, the link fails on the reference to that data.

## 7. The interpreter as oracle

`simulator/interpreter.py` executes `hc`, `arith`, `vector`, `tensor`, `memref` and `scf` directly
in Python. It needs no toolchain and runs the whole suite in seconds, so every transform is
checked by it:

1. **Same answer before and after.** `tests/test_oracle.py` runs each op through the
   interpreter before and after the whole middle end, and both results must equal the
   expected value. A FileCheck test shows which ops a pass emits, not that they compute the
   right result; this does.
2. **Independent ground truth.** Matmul and broadcasting results are compared with NumPy, not
   with a second hand-written implementation that could share a bug. Other ops use
   hand-computed expected values.
3. **`module.verify()`** after the pipeline (in tests), so invalid IR fails at the source.

Values: scalars are ints, vectors flat lists, tensors and memrefs nested row-major lists.
Tensor ops copy; memref ops mutate. The `hc` binary ops broadcast with `_broadcast_binop`,
which is written independently of the lowering. The interpreter is built to fail loudly:

- every index is bounds-checked (Python would silently wrap a negative index);
- `memref.alloc` fills with `None`, so an element nothing ever wrote fails on first arithmetic
  instead of acting like 0;
- `memref.dealloc` removes the binding, so use-after-free is a `KeyError`.

The suite has three layers:

- **FileCheck pass tests** run one pass on a small IR string and match the printed output
  against `// CHECK:` lines. They show *what* a pass emits.
- **Oracle tests** run the interpreter before and after the middle end. They show that the
  emitted code computes the right thing.
- **Compiled-path tests** (`tests/test_compiled_backend.py`) check the executable's output
  against NumPy or plain Python, since the interpreter never runs the ABI path.

Outside the suite, `full_compiler.sh` applies the same idea to whatever arguments it is given:
`hc_interpret.py` reads them the way the generated harness does and prints its result in the
harness's format, and the script fails unless the executable prints the same text.

## 8. Known limits

Each limit below exists because no current workload needs it, not by accident.

- **`i32` only.** The loader rejects any ONNX element type other than `INT32`. The scalar
  `hc.relu`/`hc.pow` lowerings hardcode `i32` constants.
- **Shapes:** rank 0 (scalar) and ranks 1–3 (tensor), with every dim a known positive size.
  The loader rejects anything else (rank 4+, symbolic dims). `hc.matmul` takes rank 2, or
  rank 3 with one leading batch dim.
- **Only `add`/`sub`/`mul` broadcast.** The `*_tensor` binary ops broadcast (§2), so a bias
  add `[M,N] + [N]` works. `hc.pow`/`hc.max`/`hc.min` are scalar-only, and `hc.matmul` needs
  equal batch dims.
- **One function, one block, one output.** The loader, bufferization and interpreter all
  assume this.
- **No overflow model.** Python ints don't wrap the way `i32` does in hardware.
- **Interpreter `scf.if`** reads the yielded value without executing the branch body. This is
  correct only while branches yield already-computed values (true for `hc.max`/`hc.min`).
- **Bufferization copies in the top-level block only** (§4). A write inside a loop body that
  would need a copy is refused.
- **DCE** only scans the function's top-level block, not loop or `if` bodies.
- **`analysis.py`**: bufferization uses `is_last_use`. The use-def and liveness printing is
  wired to a config flag, but no pass uses it, and its liveness is block-local (it doesn't see
  uses inside nested regions).
