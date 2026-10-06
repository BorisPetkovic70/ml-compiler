# ml-compiler

A small ONNX-to-native compiler built on [xDSL](https://xdsl.dev/). It imports an ONNX model
into a custom `hc` dialect, lowers it through MLIR's `arith`/`scf`/`vector`/`tensor`/`memref`
dialects, and compiles it to an executable with LLVM. A pure-Python interpreter runs the same
IR at every stage and is the test suite's correctness oracle.

**Supported:** ONNX `Add`, `Sub`, `Mul`, `MatMul`, `Relu`, `Pow`, `Max`, `Min`, `Constant`;
initializers (weights); `int32` only; one model output. Rank-0 inputs become scalars, rank-1 vectors, rank-2/3 tensors
(rank 3 = one leading batch dim). `Pow`/`Max`/`Min` are scalar-only. The loader rejects other
element types, rank 4+, symbolic dims, and operands that would need broadcasting.

```
ONNX ─▶ front_end/loader.py ─▶ hc.* ─▶ middle_end/pipeline.py ─▶ back_end/harness_gen.py ─▶ back_end/back_end.sh
                                │      lowering → bufferization    per-model C harness       mlir-opt → mlir-translate
                                │      (→ const folding → DCE,                               → llc → clang → build/<stem>_run
                                │       optional; off in hc_main)
                                └──▶ simulator/interpreter.py  (runs before and after the middle end; results must match)
```

## Setup

Requires Python 3.10. From the directory *above* `ml-compiler/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r ml-compiler/requirements-dev.txt
source .venv/bin/activate          # full_compiler.sh calls plain `python3`
cd ml-compiler
```

The compiled path additionally needs `mlir-opt`, `mlir-translate`, `llc` (an LLVM/MLIR build)
and `clang`. Nothing else does.

## Test

```bash
python -m pytest -q                          # whole suite, no LLVM needed
python -m pytest -q tests/test_lowering.py   # one file

# CI's gate: at least 85% line coverage
python -m pytest --cov=hc_dialect --cov=middle_end --cov=back_end \
  --cov=simulator --cov-report=term-missing --cov-fail-under=85 -q
```

The suite tests the compiler in two ways:

- **FileCheck pass tests** (`test_lowering.py`, `test_bufferization.py`,
  `test_constant_folding.py`, `test_dce.py`, `test_harness_gen.py`): a small IR string goes
  through one pass (or the harness generator), and the printed output, IR or C, is matched
  against `// CHECK:` lines. They show *what* a pass emits.
- **Oracle tests** (`test_oracle.py`): the interpreter runs an `hc` function before and after
  the middle end, and both results must equal a hand-computed value or NumPy's. They show that
  the emitted code is *correct*.

`test_dialect.py` checks each op's verifier, `test_loader.py` the ONNX import,
`test_interpreter.py` the interpreter itself, and `test_hc_interpret.py` the interpreter
driver's argument reading and output format.

`tests/test_compiled_backend.py` compiles and runs real executables. It is skipped unless
`mlir-opt`, `mlir-translate`, `llc` and `clang` are all on `PATH` (so it is always skipped in CI).

## Run

```bash
python front_end/build_model.py    # write the 6 sample models to build/

python hc_main.py build/matmul.onnx          # import, lower, write .mlir + harness (no LLVM)
python hc_interpret.py matmul.onnx [args...] # run in the interpreter, before and after the middle end

TOOLCHAIN_BIN_DIR=/path/to/llvm/bin ./full_compiler.sh matmul.onnx [args...]   # both + compile, run, compare
TOOLCHAIN_BIN_DIR=/path/to/llvm/bin ./run_all_models.sh                        # every sample model, fixed inputs
build/matmul_run [args...]                   # rerun a compiled model
```

`full_compiler.sh` fails unless the executable prints exactly what the interpreter computes
for the same arguments.

- **Sample models:** `score_model` (scalar), `vec_affine_relu` (vector), `matmul`,
  `chained_tensor_math` (ReLU(A@B + C)), `batched_matmul`, `dense_layer` (ReLU(X@W + B), with
  `W` and `B` stored in the model as weights).
- **Model argument:** `matmul.onnx` and `build/matmul.onnx` are equivalent; the stem (`matmul`)
  names every output file. If the file is missing, `hc_main.py` builds it — all six sample
  models are recognized by name; **any other missing name still silently builds the scalar
  score model**.
- **`TOOLCHAIN_BIN_DIR`:** directory holding `mlir-opt`/`mlir-translate`/`llc`. The default in
  `back_end/back_end.sh` is a machine-specific path; override it with the env var.
- **Args** (executable and `hc_interpret.py`): one integer per scalar, per vector lane, or per
  buffer element, in argument order. Missing args fall back to the same built-in defaults in
  both.

## Generated files

| File | Produced by | Contents |
|---|---|---|
| `build/{stem}_original.mlir` | `hc_main.py` | `hc.*` module from the loader |
| `build/{stem}_lowered.mlir` | `hc_main.py` | after lowering + bufferization |
| `back_end/{stem}_harness.c` | `hc_main.py` | C `main` matching the entry signature |
| `build/{stem}_llvm.mlir` | `back_end.sh` | `llvm` dialect, via `mlir-opt` |
| `build/{stem}_out.ll` | `back_end.sh` | LLVM IR, via `mlir-translate` |
| `build/{stem}_out.s`, `_out.o` | `back_end.sh` | assembly / object, via `llc` |
| `build/{stem}_run` | `back_end.sh` | executable, linked by `clang` |

## Layout

```
hc_dialect.py           the hc dialect: scalar, vector (*_vec) and tensor (*_tensor, matmul) ops
hc_main.py              compile driver: ONNX → hc → middle end → .mlir + harness
hc_interpret.py         interpreter driver: runs a model on given arguments, before and after the middle end
full_compiler.sh        hc_main.py + hc_interpret.py + back_end/back_end.sh, then compares the two results
run_all_models.sh       full_compiler.sh on every sample model, with fixed inputs
front_end/              loader.py (ONNX → hc), build_model.py (sample models)
middle_end/             pipeline.py, hc_lowering.py, bufferization.py, constant_folding.py,
                        dead_code_elimination.py, analysis.py
back_end/               harness_gen.py, back_end.sh
simulator/              interpreter.py
tests/                  pytest suite
docs/                   DESIGN.md, HOW_TO_ADD_AN_OP.md
doc_upload/             the xDSL tutorial (LaTeX + PDF) this project started from
```

## Further reading

- [docs/DESIGN.md](docs/DESIGN.md): architecture and the reasoning behind it.
- [docs/HOW_TO_ADD_AN_OP.md](docs/HOW_TO_ADD_AN_OP.md): checklist for a new `hc.*` op.
