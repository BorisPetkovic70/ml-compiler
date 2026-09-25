# Design

This document explains *why* the compiler is structured the way it is — the separation
between passes, the distinction between value and memory semantics, and the refusal-over-guessing
philosophy that runs through every pass. For *what* each file does and *how* to run the
pipeline, see [README.md](../README.md). For a checklist on extending the `hc` dialect, see
[HOW_TO_ADD_AN_OP.md](HOW_TO_ADD_AN_OP.md).

## 1. Why `hc.*` exists above `arith`/`scf`

ONNX describes a computation at the level of named operators over typed tensors — `MatMul`,
`Relu`, `Add` — with no notion of loops, registers, or memory. `arith`/`scf`/`vector`/`tensor`/
`memref`, on the other hand, are MLIR's low-level building blocks: arithmetic on SSA values,
structured control flow, and (eventually) memory. Lowering straight from ONNX into those
low-level dialects would collapse two genuinely separate decisions into one pass: *what* the
model computes, and *how* that computation is realized as loops and memory.

`hc` is the layer in between. It is ONNX-shaped (one op per ONNX node, one operand per ONNX
input) but already xDSL/MLIR IR, so it can be verified, printed, and manipulated with the same
tools as everything downstream. Splitting the frontend this way means the loader
(`front_end/loader.py`) only has to answer "which `hc.*` op does this ONNX node become," and
every question about *how* that op executes is answered later, uniformly, by the lowering pass.

## 2. Three value categories, and why they diverge

The dialect (and every pass downstream of it) treats scalars, vectors, and tensors as
fundamentally different kinds of value, not as one generic "tensor of rank N":

- **Scalars** (`i32`) map directly onto a CPU register and onto `arith`'s scalar ops. There is
  nothing to lower — `hc.add` becomes `arith.addi` and that's the whole transformation.
- **Vectors** (`vector<Nxi32>`) map onto a SIMD register. `arith`'s ops already operate
  element-wise on vector operands when the types match, so most vector lowering is likewise a
  direct substitution; the one wrinkle is a scalar operand that needs broadcasting into a vector
  lane count (`hc.mul_vec`, below).
- **Tensors** (`tensor<MxNxi32>`) have **no hardware counterpart**. A tensor cannot live in a
  register — it needs to be visited element-by-element, which means it needs a loop, and a loop
  needs somewhere to read and write, which eventually means memory. This is the fact that
  produces the rest of this document: `hc.matmul` is the first op in the dialect whose lowering
  has to *introduce* structure (loops) that wasn't in the source op at all.

This is also why the dialect defines separate op families (`HCAdd`/`HCAddVec`, `HCMul`/
`HCMulVec`/`HCMulVecVec`) instead of one polymorphic op per operation: each family's `verify_()`
enforces different invariants (scalar equality vs. vector shape equality vs. rank-2 shape
compatibility), and conflating them would mean one verifier trying to enforce three unrelated
sets of rules.

## 3. Value semantics vs. memory semantics

This is the central distinction in the whole middle end, and it is easiest to see by comparing
two operations that look similar but are not:

- **`tensor.insert`** (used by `hc.matmul`'s lowering) has **value semantics**: it takes a
  tensor and a new element value and returns a *brand-new* tensor with that one element
  changed. The original tensor is completely unaffected — anyone still holding a reference to it
  sees the old, unchanged value. This is why the lowered matmul loop nest threads the
  accumulating result tensor through `iter_args`: each loop iteration's `tensor.insert` produces
  a new tensor value, which must explicitly be carried into the next iteration, because nothing
  is being mutated in place.
- **`memref.store`** (what `tensor.insert` becomes after bufferization) has **memory
  semantics**: it writes into a buffer in place. Two different SSA values that happen to point
  at the same underlying buffer will both observe a write made through either one.

Bufferization (`middle_end/bufferization.py`) is the pass that converts from the first world to
the second. Its single most visible effect is that the `i`/`j` loops in the matmul nest, which
previously had to carry the result tensor through `iter_args` because every `tensor.insert`
produced a new value, no longer carry anything at all — the `memref.store` that replaced
`tensor.insert` mutates one buffer directly, so there is nothing left to thread between
iterations. (The innermost `k` loop still carries a scalar accumulator; that is a legitimate
register-level reduction, not a memory buffer, and is unaffected by bufferization.)

This substitution — a copying `tensor.insert` for an in-place `memref.store` — is only
correct when the old tensor value is provably dead afterward. If some other, still-live
reference to the "old" tensor exists, mutating a shared buffer in place would silently change
what that other reference observes. Section 4 covers how the pass handles this.

## 4. The refusal philosophy

**A pass that cannot prove a transformation is safe raises an error, rather than emitting
code that runs but produces a wrong answer.** This rule shows up in every pass in the
pipeline, and it is worth stating explicitly because it is the single most consistent design
decision across the whole codebase.

Concretely, this is where it applies:

- **Bufferization** (`middle_end/bufferization.py::apply_bufferization`) refuses, rather than
  silently mutating a buffer that might still be needed elsewhere, in three situations: a tensor
  fed into an insert (or a loop's `iter_arg` init) has more than one use; an insert targets a
  function argument (there is no caller to prove the original value is no longer needed, since
  the pipeline only ever has one function per module); or a loop yields a different buffer than
  the one it carries. None of these fire on `hc.matmul`'s own generated nest — verified
  empirically that they don't even fire when two `hc.matmul`s are chained and share an
  intermediate tensor — but they exist so that a *future*, differently-shaped lowering is caught
  loudly instead of silently miscompiled.
- **The harness generator** (`back_end/harness_gen.py`) refuses to generate a C harness for a
  rank-3+ memref, a non-integer element type, a function that takes a memref argument but
  returns something other than a memref, or a signature that mixes a vector argument with a
  memref one. Each of these is a real, distinct C ABI shape — refusing them means a signature
  the generator doesn't understand produces an error at generation time, not a harness that
  compiles, links, and prints a wrong number.
- **The interpreter** (`simulator/interpreter.py`) bounds-checks every tensor/memref index
  explicitly (Python lists would otherwise silently interpret a negative index as counting from
  the end), and fills a freshly allocated `memref.alloc` buffer with `None` rather than `0`. Any
  cell that some bug leaves unwritten then raises a `TypeError` the moment it is used in
  arithmetic, instead of silently participating in a computation as a plausible-looking zero.
  `memref.dealloc` similarly deletes the value's binding outright, so a later use of a freed
  buffer hits the same `KeyError` any other missing-value bug would — no bespoke
  use-after-free detector needed, just not hiding the mistake.

Every one of these refusal sites has a negative test asserting the error actually fires (see
[HOW_TO_ADD_AN_OP.md](HOW_TO_ADD_AN_OP.md) for the pattern to follow when adding a new one).

## 5. Verification strategy

The pipeline has two independent ways to run a model: compile it all the way to a native
executable through LLVM, or interpret the IR directly in Python
(`simulator/interpreter.py`). The interpreter is not a fallback or a toy — it is the
**correctness oracle** the entire test suite is built on, precisely because it needs no LLVM
toolchain and runs in well under a second for the whole suite.

The verification discipline has stayed consistent across every pass added to this pipeline:

1. **Semantic equality before and after a transform.** Every operator is tested by running the
   *same* IR through the interpreter before lowering and after lowering (and, for matmul, after
   bufferization too) and asserting identical results. This is what actually proves a lowering
   is correct — a structural check can confirm the right *kind* of operations appear, but only a
   semantic check can catch a lowering that wires them together incorrectly (for example, a
   loop nest that produces the right operation types but reads from the wrong index).
2. **An independent ground truth for anything beyond hand-checkable arithmetic.** `hc.matmul`'s
   correctness is checked against NumPy's `@` operator, not against a second hand-written
   reference implementation that could share the same bug as the first.
3. **`module.verify()` after every pass.** `MiddleEndPipeline.apply_passes()` does not verify
   its own output; the test suite's `conftest.lower()` helper calls `module.verify()`
   immediately after running the pipeline, specifically so a structurally invalid lowering fails
   loudly at the point of the bug instead of reaching the interpreter and producing a
   wrong-but-plausible number.


## 6. The ABI boundary (compiled backend)

A single static C harness cannot declare a generic `extern` signature that works for every
model, because the argument and result types genuinely vary (scalar-only, vector-only, memref).
`back_end/harness_gen.py` instead generates a matching harness per model, and — because the two
possible ABI shapes are genuinely different calling conventions — it picks between two
generators based on whether a memref appears anywhere in the signature:

- **Register ABI** (no memref anywhere): calls the compiled function directly. Scalars pass as
  `int`; vectors pass as a GCC `vector_size` value, laid out so they travel in a SIMD register.
- **C-interface ABI** (any memref in the signature): calls `_mlir_ciface_<fn>`, the wrapper that
  MLIR's `--llvm-request-c-wrappers` pass emits. This was confirmed against the actual generated
  LLVM IR rather than assumed from documentation: scalars and vectors are *still* passed by
  value in this convention, exactly as in the register ABI; only a memref becomes a pointer to a
  descriptor struct (`{allocated, aligned, offset, sizes[], strides[]}`); and a memref *result*
  becomes a hidden leading output parameter rather than an ordinary return value, because the
  descriptor struct is too large to return directly. The callee allocates the result buffer
  (`memref.alloc` lowers to `malloc`), so the harness is responsible for freeing it.

Because bufferization can turn a tensor-typed signature into a memref one, the harness must be
generated *after* the middle end runs, not before — `hc_main.py` reflects this ordering.

## 7. Known limits, and why each one is a limit rather than a bug

- **Only rank-1 and rank-2 shapes are supported** anywhere memref/tensor types appear (the
  dialect's own verifiers, the harness generator, bufferization). There is currently no
  workload that needs rank-3+, and generalizing the loop-nest-building and descriptor-struct
  code without a concrete test case to validate against would be speculative.
- **Only `i32`.** No floating-point element type exists anywhere in the pipeline. This is a
  scope decision, not an oversight — the whole verification chain (interpreter, NumPy
  comparison, C ABI descriptor structs) currently assumes integer arithmetic.
- **Only single-block function bodies.** Both the interpreter and bufferization assume a
  function's body is one straight-line block (with structured control flow — `scf.for`/`scf.if`
  — nested inside it, which is fine). Nothing in the dialect currently produces unstructured
  control flow, so this has never been exercised.
- **No integer overflow modeling.** The interpreter uses Python's arbitrary-precision integers
  throughout, so an `i32` computation that would wrap around on real hardware simply doesn't in
  simulation. This is invisible for the shapes and values currently used in tests.
- **`scf.if` in the interpreter only reads the yielded value; it does not execute the branch's
  operations.** This is harmless today because `hc.max`/`hc.min`'s lowering (the only source of
  `scf.if` in the pipeline) yields an already-computed outer value directly from each branch —
  but it would read a stale value if some future op's lowering computed something *inside* a
  branch.
- **Bufferization refuses rather than copies** when it cannot prove an in-place mutation safe
  (see Section 4). A production version of this pass would resolve each tensor value to a "root"
  buffer — cheap here, since every value in this IR has exactly one static definition, making it
  an exact def-chain walk rather than an approximate alias analysis — and insert a `memref.copy`
  only at the specific point where a write cannot be proven safe. This was designed but not
  built, since nothing in the current pipeline exercises the aliasing cases it would handle.
- **The harness generator refuses to mix a vector argument with a memref one.** This is a real,
  distinct ABI shape (confirmed against generated LLVM IR: a vector argument stays by value even
  in the C-interface calling convention), but nothing in the current dialect produces a
  signature that combines the two, so supporting it would be unvalidated by any real model.
- **The register-ABI's `vector_size` calling convention has a width ceiling.** It is clean for
  small vector widths that fit in a SIMD register; a large `N` would need the memref/pointer
  calling convention instead. That convention now *exists* (it was built for tensors), but
  nothing currently routes an oversized vector through it — the harness generator dispatches
  purely on whether a `MemRefType` appears in the signature, and a large vector is still a
  `VectorType`. Wiring "vector too wide for registers" onto the existing memref path is a
  smaller change than it would have been before the compiled backend existed, but it hasn't
  been done.
