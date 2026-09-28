"""End-to-end compiled-backend tests: build a real model, run it all the way through
the LLVM/MLIR toolchain to a native executable, and check the executable's own stdout
against an independent oracle (plain Python or NumPy) -- not the project's own
interpreter, since the whole point of these tests is to check the compiled ABI path,
which the interpreter never exercises. String-checking the generated harness source
(see test_harness_gen.py) can confirm the descriptor layout looks right; only actually
compiling and running it can confirm the ABI is right.

Skipped whenever the LLVM/MLIR toolchain isn't on PATH, so the rest of the suite stays
toolchain-free -- these are the only tests in the project that spawn a subprocess.
"""
import re
import shutil
import subprocess

import pytest

from hc_main import build_context
from front_end.loader import import_onnx_to_hc_module
from front_end.build_model import (
    build_score_model, build_vec_affine_relu_model, build_matmul_model,
)
from middle_end.pipeline import MiddleEndPipeline, MiddleEndPipelineConfig
from back_end.harness_gen import write_harness

_TOOLCHAIN = ("mlir-opt", "mlir-translate", "llc", "clang")
pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in _TOOLCHAIN),
    reason="LLVM/MLIR toolchain not found on PATH",
)


def _compile_and_run(tmp_path, onnx_path, run_args):
    """Load, lower (folding+DCE+bufferization all on, the real pipeline's defaults),
    write a harness, and compile it to a native executable via the exact same
    mlir-opt/mlir-translate/llc/clang pass list as back_end/back_end.sh -- replicated
    here with PATH-resolved tools rather than by invoking that script, since its
    TOOLCHAIN_BIN_DIR is a hardcoded, machine-specific path (see CLAUDE.md). Runs the
    executable with run_args and returns its stdout."""
    ctx = build_context()
    module = import_onnx_to_hc_module(ctx, str(onnx_path), fn_name="my_func")

    cfg = MiddleEndPipelineConfig(
        apply_lowering=True, apply_bufferization=True,
        apply_constant_folding=True, apply_dce=True,
        run_analysis=False, debug_mode=False,
    )
    MiddleEndPipeline(cfg).apply_passes(module)
    module.verify()

    lowered = tmp_path / "lowered.mlir"
    lowered.write_text(str(module))
    harness = tmp_path / "harness.c"
    write_harness(module, harness, func_name="my_func")

    def tool(name: str) -> str:
        return shutil.which(name)

    llvm_mlir = tmp_path / "llvm.mlir"
    subprocess.run(
        [tool("mlir-opt"), str(lowered),
         "--llvm-request-c-wrappers", "--expand-strided-metadata",
         "--finalize-memref-to-llvm", "--lower-affine",
         "-convert-vector-to-llvm", "-convert-scf-to-cf", "-convert-cf-to-llvm",
         "-convert-arith-to-llvm", "-convert-func-to-llvm", "-reconcile-unrealized-casts",
         "-o", str(llvm_mlir)],
        check=True, capture_output=True, text=True,
    )
    out_ll = tmp_path / "out.ll"
    subprocess.run(
        [tool("mlir-translate"), str(llvm_mlir), "-mlir-to-llvmir", "-o", str(out_ll)],
        check=True, capture_output=True, text=True,
    )
    out_o = tmp_path / "out.o"
    subprocess.run(
        [tool("llc"), "-O2", "-filetype=obj", str(out_ll), "-o", str(out_o)],
        check=True, capture_output=True, text=True,
    )
    run_bin = tmp_path / "run"
    subprocess.run(
        [tool("clang"), str(harness), str(out_o), "-o", str(run_bin)],
        check=True, capture_output=True, text=True,
    )

    result = subprocess.run(
        [str(run_bin), *[str(a) for a in run_args]],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _parse_ints(text: str) -> list[int]:
    return [int(v) for v in re.findall(r"-?\d+", text)]


def test_scalar_abi_compiles_and_matches_independent_oracle(tmp_path):
    """Register ABI, scalar-only signature: calls the compiled function directly,
    scalars passed as plain int. score = relu(min(max(alpha*abs(x-mu)**p, lo), hi))."""
    onnx_path = tmp_path / "score.onnx"
    build_score_model(str(onnx_path))

    x = 25
    mu, alpha, p, lo, hi = 40, 2, 2, 0, 1000
    abs_d = max(x - mu, mu - x)
    e = abs_d ** p
    s = alpha * e
    lower = max(s, lo)
    clamped = min(lower, hi)
    expected = max(clamped, 0)

    stdout = _compile_and_run(tmp_path, onnx_path, [x])
    assert _parse_ints(stdout) == [expected]


def test_vector_abi_compiles_and_matches_independent_oracle(tmp_path):
    """Register ABI, vector signature: vectors passed as GCC vector_size values.
    y = relu((a*x + b) * w), w = [7,2,3,5]; b chosen with a negative lane so relu
    actually clamps something."""
    onnx_path = tmp_path / "vec_affine_relu.onnx"
    build_vec_affine_relu_model(str(onnx_path), n=4)

    x = [1, 2, 3, 4]
    a = 3
    b = [-100, -5, 0, 2]
    w = [7, 2, 3, 5]
    t = [a * xi + bi for xi, bi in zip(x, b)]
    p = [ti * wi for ti, wi in zip(t, w)]
    expected = [max(v, 0) for v in p]

    stdout = _compile_and_run(tmp_path, onnx_path, [*x, a, *b])
    assert _parse_ints(stdout) == expected


def test_memref_abi_compiles_and_matches_independent_oracle(tmp_path):
    """C-interface ABI (_mlir_ciface_<fn>): memrefs passed as pointers to descriptor
    structs, the memref result as a hidden leading out-parameter. C = A @ B, via
    bufferization -- the ABI path only a tensor/memref-bearing model exercises."""
    np = pytest.importorskip("numpy")
    onnx_path = tmp_path / "matmul.onnx"
    m, k, n = 2, 3, 4
    build_matmul_model(str(onnx_path), m=m, k=k, n=n)

    rng = np.random.default_rng(seed=7)
    a = rng.integers(-5, 6, size=(m, k))
    b = rng.integers(-5, 6, size=(k, n))
    expected = np.matmul(a, b).tolist()

    args = [*a.flatten().tolist(), *b.flatten().tolist()]
    stdout = _compile_and_run(tmp_path, onnx_path, args)
    actual = [_parse_ints(row) for row in stdout.splitlines() if row.strip()]
    assert actual == expected
