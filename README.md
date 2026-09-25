# ml-compiler

An [xDSL](https://xdsl.dev/)-based compiler that takes a small ONNX model and either
interprets it directly in Python or compiles it all the way to a native executable through
LLVM. Built as a learning project extending the pattern (source in `doc_upload/`, committed here as reference).

For the architectural reasoning behind the pipeline's design, see
[docs/DESIGN.md](docs/DESIGN.md). For a checklist on adding a new operator, see
[docs/HOW_TO_ADD_AN_OP.md](docs/HOW_TO_ADD_AN_OP.md).

## Pipeline

```
                              ┌─────────────────────────────────────┐
                              │   simulator/interpreter.py           │
                              │   (parallel, toolchain-free path —   │
                              │   the correctness oracle every       │
                              │   pass is checked against)           │
                              └───────────────┬───────────────────────┘
                                              │ same hc.* module, at every stage
ONNX model ──▶ front_end/loader.py ──▶ hc.* dialect module
                                              │
                                              ▼
                                  middle_end/pipeline.py
                        (lowering → bufferization → constant folding → DCE)
                                              │
                                              ▼
                                  back_end/harness_gen.py
                          (generates a C harness matching the model's
                           actual entry signature — see docs/DESIGN.md §6)
                                              │
                                              ▼
                                    back_end/back_end.sh
                       mlir-opt → mlir-translate → llc → clang → native executable
```

The interpreter is not a fallback for when the compiler doesn't work — it is run on every model
at every stage (before lowering, after lowering, after bufferization) specifically to catch a
transform that changes the model's meaning, and it is what the entire pytest suite is verified
against.

## Quickstart

```bash
# from xdsl-playground/
python3 -m venv .venv
.venv/bin/pip install -r ml-compiler/requirements-dev.txt xdsl==0.70.0 onnx==1.22.0

cd ml-compiler
../.venv/bin/python -m pytest -q          # run the test suite (no LLVM toolchain needed)

./full_compiler.sh matmul.onnx            # build a model, compile it, run the executable
```

## Calling conventions

- **The model argument is a name, not necessarily a literal path.** `./full_compiler.sh
  matmul.onnx` and `./full_compiler.sh build/matmul.onnx` both work and produce identical
  results — the driver (`hc_main.py`) and the backend script (`back_end/back_end.sh`) each
  derive the model's "stem" (the name used for every generated file) from this argument by
  stripping the `.onnx` suffix and any directory component (`Path(...).stem` on the Python
  side, `basename "${VAR%.onnx}"` on the bash side), so either form resolves to the same stem.
  If the file isn't found where given, `hc_main.py` also checks `build/<name>.onnx` before
  falling back to building it — see below — so a bare name works even when the actual file
  lives in `build/`.
- **Unknown model names build the score model by default.** If the named `.onnx` file doesn't
  exist yet (checked both at the given path and in `build/`), `hc_main.py` builds it via
  `front_end/build_model.py`, dispatching on the file's name — `matmul.onnx` and
  `vec_affine_relu.onnx` are recognized by name; anything else falls back to the default
  scalar score model.
- **`back_end/back_end.sh`'s `TOOLCHAIN_BIN_DIR` is machine-specific.** It points at wherever
  your own local LLVM/MLIR build lives (`mlir-opt`, `mlir-translate`, `llc`). Edit that one line
  for your machine; it is intentionally not portable and not something a working pipeline
  depends on being any particular path.
- **`build/{stem}_run [args...]`** runs a previously compiled executable directly. For a
  scalar/vector model, `args` are read positionally (one `int` per scalar, N per vector); for a
  memref (tensor) model, one argument fills one buffer element, in argument order. In both
  cases, any arguments not supplied fall back to built-in default values — you don't need to
  supply all of them.

## What each generated file is

Running `full_compiler.sh <model>` produces, under `build/` (and `back_end/` for the harness),
in this order:

| File | Produced by | What it is |
|---|---|---|
| `{stem}_original.mlir` | `hc_main.py` | The `hc.*` dialect module, straight from the loader |
| `{stem}_lowered.mlir` | `hc_main.py` | After the middle-end pipeline (lowering, bufferization, folding, DCE) |
| `{stem}_harness.c` | `hc_main.py` | Generated C harness matching this model's actual entry signature |
| `{stem}_llvm.mlir` | `back_end.sh` | After `mlir-opt`'s lowering to the `llvm` dialect |
| `{stem}_out.ll` | `back_end.sh` | LLVM IR, via `mlir-translate` |
| `{stem}_out.s` / `{stem}_out.o` | `back_end.sh` | Assembly and object file, via `llc` |
| `{stem}_run` | `back_end.sh` | The final native executable, linked via `clang` |

## Repo layout

```
hc_dialect.py                  hc dialect: scalar + vector + tensor (hc.matmul) op definitions
hc_main.py                     driver: ONNX -> hc module -> middle-end -> .mlir + harness.c
full_compiler.sh                hc_main.py + back_end/back_end.sh
pyproject.toml                 pytest config (pythonpath=["."], testpaths=["tests"])
requirements-dev.txt           dev deps for the test suite: pytest, pytest-cov, numpy
tests/                         pytest suite
front_end/
  build_model.py               ONNX model builders: score, vec_affine_relu, matmul (fixtures)
  loader.py                    import_onnx_to_hc_module: unified scalar/vector/matmul ONNX -> hc.*
middle_end/
  pipeline.py                  MiddleEndPipeline / MiddleEndPipelineConfig / _apply_pass
  hc_lowering.py                LowerHCPattern: hc.* -> arith./scf./vector./tensor.
  bufferization.py             apply_bufferization: hand-written tensor -> memref rewrite
  constant_folding.py          FoldArithInts: fold arith.addi/muli/subi/maxsi of constants
  dead_code_elimination.py     apply_dce: iterative fixpoint DCE on arith.* results
  analysis.py                  use-def report + block-local liveness (currently unused by any pass)
back_end/
  back_end.sh                  mlir-opt -> mlir-translate -> llc -> clang -> run
  harness_gen.py                generate_harness_c / write_harness: per-model C harness
simulator/
  interpreter.py               pure-Python interpreter (the pytest oracle; see "Pipeline" above)
doc_upload/                    tutorial LaTeX source + main.pdf (committed as reference)
docs/                          DESIGN.md, HOW_TO_ADD_AN_OP.md
```

## Useful commands

```bash
# Run the test suite
../.venv/bin/python -m pytest -q                          # or -v for per-test names
../.venv/bin/python -m pytest tests/test_lowering.py -v   # one file

# Coverage report -- check for uncovered verify_()/raise branches
../.venv/bin/python -m pytest --cov=hc_dialect --cov-report=term-missing -q

# Run the driver alone: builds/loads a model, writes .mlir + harness.c, runs interpreter checks
python3 hc_main.py [model.onnx]

# Full pipeline: driver + backend compile/link/run (edit TOOLCHAIN_BIN_DIR first -- see above)
./full_compiler.sh [model.onnx]

# Run a previously compiled executable directly
build/{stem}_run [args...]
```
