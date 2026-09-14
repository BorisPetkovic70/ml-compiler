from xdsl.dialects import arith, builtin, scf, vector
from xdsl.ir import Block, Operation, Region
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter
)

def _const_i32(value: int) -> arith.ConstantOp:
    return arith.ConstantOp.from_int_and_width(value, 32)


# -----------------------------------------------------------------------------
#  Vector lowering helpers (ported from step13_vec_operators.py)
# -----------------------------------------------------------------------------
def _as_int(x) -> int:
    """Convert xDSL int-like objects (IntAttr, nested attrs) to a python int."""
    if isinstance(x, int):
        return x
    if hasattr(x, "data"):
        return int(x.data)
    if hasattr(x, "value") and hasattr(x.value, "data"):
        return int(x.value.data)
    return int(x)


def _vec_num_elements(vec_type: builtin.VectorType) -> int:
    """Total number of elements in a vector type (product of its shape dims)."""
    shape = getattr(vec_type, "shape", None)
    if shape is None:
        raise RuntimeError(f"Vector type has no 'shape' attribute: {vec_type}")
    n = 1
    for d in shape:
        n *= _as_int(d)
    return n


def _const_zero_like(like_type) -> arith.ConstantOp:
    """Create a constant 0 with the same type as `like_type` (scalar or vector)."""
    if isinstance(like_type, builtin.VectorType):
        count = _vec_num_elements(like_type)
        dense = builtin.DenseIntOrFPElementsAttr.from_list(like_type, [0] * count)
        return arith.ConstantOp(dense)
    # scalar integer fallback
    return _const_i32(0)


# -----------------------------------------------------------------------------
#  Lowering pattern: hc -> arith
# -----------------------------------------------------------------------------
class LowerHCPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Debug: uncomment if you want to see traversal
        # print("VISIT:", op.name)

        if op.name not in (
            "hc.add", "hc.mul", "hc.sub", "hc.relu", "hc.pow", "hc.max", "hc.min",
            "hc.add_vec", "hc.sub_vec", "hc.mul_vec", "hc.mul_vec_vec", "hc.relu_vec",
        ):
            return

        # ---------------- hc.add ----------------
        if op.name == "hc.add":
            lhs, rhs = op.operands
            new_op = arith.AddiOp(lhs, rhs)
            rewriter.replace_op(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.mul ----------------
        if op.name == "hc.mul":
            lhs, rhs = op.operands
            new_op = arith.MuliOp(lhs, rhs)
            rewriter.replace_op(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.sub ----------------
        if op.name == "hc.sub":
            lhs, rhs = op.operands
            new_op = arith.SubiOp(lhs, rhs)
            rewriter.replace_op(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.relu ----------------
        if op.name == "hc.relu":
            x = op.operands[0]
            c0 = _const_i32(0)

            if hasattr(arith, "MaxSIOp"):
                maxop = arith.MaxSIOp(x, c0.result)
                rewriter.replace_op(
                    op,
                    new_ops=[c0, maxop],
                    new_results=[maxop.result],
                    safe_erase=True,
                )
                return
            else:
                raise RuntimeError(
                    "Cannot lower hc.relu: arith.MaxSIOp needed."
                )
        # ---------------- hc.pow ----------------
        if op.name == "hc.pow":
            base, exp = op.operands
            ty = op.results[0].type  # integer type

            iv_ty = builtin.IndexType()

            # loop-control constants must be index
            c0 = arith.ConstantOp.from_int_and_width(0, iv_ty)
            c1 = arith.ConstantOp.from_int_and_width(1, iv_ty)

            # exponent is i32 -> cast to index for scf.for upper bound
            exp_idx = arith.IndexCastOp(exp, iv_ty)

            # init accumulator = 1
            init = arith.ConstantOp.from_int_and_width(1, 32)

            # scf.for body arguments: (iv: index, acc: i32)
            body = Block(arg_types=[iv_ty, ty])
            iv, acc = body.args

            mul = arith.MuliOp(acc, base)
            body.add_ops([mul, scf.YieldOp(mul.result)])

            loop = scf.ForOp(
                lb=c0.result,          # lower bound
                ub=exp_idx.result,     # upper bound
                step=c1.result,        # step
                iter_args=[init.result],
                body=Region(body),
            )

            # scf.ForOp returns the iter_args results
            rewriter.replace_op(
                op,
                new_ops=[c0, c1, exp_idx, init, loop],
                new_results=[loop.results[0]],
                safe_erase=True,
            )
            return
        # ---------------- hc.max ----------------
        if op.name == "hc.max":
            lhs, rhs = op.operands
            ty = op.results[0].type

            cmp = arith.CmpiOp(lhs, rhs, "sgt")  # signed greater-than

            then_block = Block(arg_types=[])
            then_block.add_ops([scf.YieldOp(lhs)])

            else_block = Block(arg_types=[])
            else_block.add_ops([scf.YieldOp(rhs)])

            if_op = scf.IfOp(
                cmp.result,
                [ty],
                Region(then_block),
                Region(else_block),
            )

            rewriter.replace_op(
                op,
                new_ops=[cmp, if_op],
                new_results=[if_op.results[0]],
                safe_erase=True,
            )
            return
        if op.name == "hc.min":
            lhs, rhs = op.operands
            ty = op.results[0].type

            cmp = arith.CmpiOp(lhs, rhs, "slt")  # signed less-than

            then_block = Block(arg_types=[])
            then_block.add_ops([scf.YieldOp(lhs)])

            else_block = Block(arg_types=[])
            else_block.add_ops([scf.YieldOp(rhs)])

            if_op = scf.IfOp(
                cmp.result,
                [ty],
                Region(then_block),
                Region(else_block),
            )

            rewriter.replace_op(
                op,
                new_ops=[cmp, if_op],
                new_results=[if_op.results[0]],
                safe_erase=True,
            )
            return

        # ==========================================================
        #  Vector operations
        #  arith.* ops operate element-wise on vectors when the
        #  operand/result types match.
        # ==========================================================

        # ---------------- hc.add_vec ----------------
        if op.name == "hc.add_vec":
            lhs, rhs = op.operands
            new_op = arith.AddiOp(lhs, rhs)
            rewriter.replace_op(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.sub_vec ----------------
        if op.name == "hc.sub_vec":
            lhs, rhs = op.operands
            new_op = arith.SubiOp(lhs, rhs)
            rewriter.replace_op(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.mul_vec (scalar * vector -> vector) ----------------
        if op.name == "hc.mul_vec":
            scalar, vec = op.operands

            # arith.muli needs matching types, so broadcast the (possibly runtime)
            # scalar to vec's vector type.
            bcast = vector.BroadcastOp(scalar, vec.type)
            mul = arith.MuliOp(vec, bcast.vector)
            rewriter.replace_op(
                op,
                new_ops=[bcast, mul],
                new_results=[mul.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.mul_vec_vec (vector * vector -> vector) ----------------
        if op.name == "hc.mul_vec_vec":
            lhs, rhs = op.operands
            new_op = arith.MuliOp(lhs, rhs)
            rewriter.replace_op(
                op,
                new_ops=[new_op],
                new_results=[new_op.result],
                safe_erase=True,
            )
            return

        # ---------------- hc.relu_vec ----------------
        if op.name == "hc.relu_vec":
            x = op.operands[0]
            c0 = _const_zero_like(x.type)  # dense-zero vector of x's type

            if hasattr(arith, "MaxSIOp"):
                maxop = arith.MaxSIOp(x, c0.result)
                rewriter.replace_op(
                    op,
                    new_ops=[c0, maxop],
                    new_results=[maxop.result],
                    safe_erase=True,
                )
                return
            else:
                raise RuntimeError("Cannot lower hc.relu_vec: arith.MaxSIOp needed.")



