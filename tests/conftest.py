"""Shared test helpers: parsing IR text, FileCheck, building, running and
lowering modules. Nothing here needs the LLVM toolchain.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.parser import Parser
from xdsl.dialects.builtin import (
    Builtin, ModuleOp, i32, IntegerType, VectorType, TensorType,
    DenseIntOrFPElementsAttr,
)
from xdsl.dialects import func, arith, memref, scf, tensor, vector
from hc_dialect import HiCompiler
from middle_end.pipeline import MiddleEndPipeline, MiddleEndPipelineConfig
from simulator.interpreter import Interpreter


def _make_ctx() -> Context:
    c = Context()
    for d in (Builtin, func.Func, arith.Arith, scf.Scf, tensor.Tensor,
              memref.MemRef, vector.Vector, HiCompiler):
        c.load_dialect(d)
    return c


_CTX = _make_ctx()


@pytest.fixture(scope="session")
def ctx() -> Context:
    return _CTX


def parse(ir: str) -> ModuleOp:
    """Parses MLIR text into a module. `hc` ops are written in generic form,
    e.g. `%c = "hc.add"(%a, %b) : (i32, i32) -> i32`."""
    return Parser(_CTX, ir).parse_module()


def filecheck(output, checks: str) -> None:
    """Runs FileCheck: matches the `// CHECK:` lines in `checks`, in order,
    against `output` (a module or text). Fails the test with FileCheck's
    message on a mismatch."""
    text = str(output)
    with tempfile.TemporaryDirectory() as d:
        check_file = Path(d) / "checks"
        check_file.write_text(checks)
        proc = subprocess.run(
            [sys.executable, "-m", "filecheck", str(check_file)],
            input=text, capture_output=True, text=True,
        )
    if proc.returncode != 0:
        pytest.fail(f"FileCheck failed:\n{proc.stderr}\n--- output ---\n{text}",
                    pytrace=False)


def vec_ty(n: int, width: int = 32) -> VectorType:
    elem = IntegerType(width)
    try:
        return VectorType([n], elem)
    except TypeError:
        return VectorType(elem, [n])


def tensor_ty(*dims: int, width: int = 32) -> TensorType:
    """TensorType<...xi32> from 2 dims (MxN) or 3 dims (a leading batch dim,
    BxMxN)."""
    return TensorType(IntegerType(width), list(dims))


def const_i32(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(value, 32)


def const_vec(values: list[int]) -> arith.ConstantOp:
    ty = vec_ty(len(values))
    return arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(ty, values))


def build_module(arg_types, body, name: str = "my_func") -> ModuleOp:
    """body(block_args) -> (ops_to_insert, return_value)."""
    block = Block(arg_types=list(arg_types))
    ops, ret = body(list(block.args))
    for op in ops:
        block.add_op(op)
    block.add_op(func.ReturnOp(ret))
    fn = func.FuncOp(name, (list(arg_types), [ret.type]), region=Region(block))
    return ModuleOp(ops=[fn])


def run(module: ModuleOp, args=(), func_name: str = "my_func"):
    return Interpreter().run_module(module, args=list(args), func_name=func_name)


def lower(module: ModuleOp, fold: bool = False, dce: bool = False, bufferize: bool = False) -> ModuleOp:
    cfg = MiddleEndPipelineConfig(
        apply_lowering=True,
        apply_bufferization=bufferize,
        apply_constant_folding=fold,
        apply_dce=dce,
        run_analysis=False,
        debug_mode=False,
    )
    MiddleEndPipeline(cfg).apply_passes(module)
    # apply_passes() never calls verify() itself -- catch structurally invalid
    # IR here, at the point of lowering, rather than downstream (or never).
    module.verify()
    return module


def _entry_fn(module: ModuleOp, func_name: str = "my_func") -> func.FuncOp:
    for top_block in module.body.blocks:
        for op in top_block.ops:
            if isinstance(op, func.FuncOp) and op.sym_name.data == func_name:
                return op
    raise RuntimeError(f"Function not found: {func_name}")


def entry_op_names(module: ModuleOp, func_name: str = "my_func") -> list[str]:
    fn = _entry_fn(module, func_name)
    return [op.name for op in list(fn.body.blocks)[0].ops]


def find_op(module: ModuleOp, op_name: str, func_name: str = "my_func"):
    """First op with the given name in the entry function's body, or None."""
    fn = _entry_fn(module, func_name)
    for op in list(fn.body.blocks)[0].ops:
        if op.name == op_name:
            return op
    return None


def find_ops(module: ModuleOp, op_name: str, func_name: str = "my_func") -> list:
    """Every op with the given name anywhere under the entry function, in
    pre-order -- unlike find_op, this reaches ops nested inside scf.for bodies."""
    fn = _entry_fn(module, func_name)
    return [op for op in fn.walk() if op.name == op_name]
