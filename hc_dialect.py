"""The `hc` dialect: ONNX-shaped integer ops. See docs/DESIGN.md Sections 1-2.

`hc.add/sub/mul/max/min/pow/relu` are element-wise. An operand is an integer
or a tensor of rank >= 1, and the binary ops broadcast their operands with
NumPy's rules (`broadcast_shape`). The result is an integer when every
operand is, else a tensor. `hc.matmul` multiplies the last two dims of
tensors of rank >= 2. Any leading batch dims must be equal on both operands:
it does not broadcast.

`TensorInt` constrains only the element type. Every rule relating the types
or shapes of an op's operands and result is enforced by that op's
`verify_()`, which raises `ValueError`.

An op belongs to the dialect only once it is listed in the `HiCompiler` tuple
at the bottom of this file (checklist: docs/HOW_TO_ADD_AN_OP.md).
"""
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    base,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.dialects.builtin import IntegerType, TensorType

# -----------------------------
# Types and shapes
# -----------------------------
# "Tensor of integers" type constraint (any element width; rank checked in verify_)
TensorInt = TensorType.constr(element_type=IntegerType)

# A value of an element-wise op: a scalar, or a tensor of integers.
ScalarOrTensorInt = base(IntegerType) | TensorInt


def _dim(int_attr) -> int:
    return int_attr.data


def shape_of(ty) -> list[int]:
    """Returns the dims of a shaped type, or [] for an integer type."""
    if isinstance(ty, IntegerType):
        return []
    return [_dim(d) for d in ty.shape]


def broadcast_shape(a: list[int], b: list[int]) -> list[int]:
    """Returns the shape that shapes `a` and `b` broadcast to, by NumPy's
    rules.

    The shapes are aligned at their last dim, and the shorter one is padded
    with 1s on the left. Each pair of dims must be equal, or one of them 1;
    the result takes the larger. Raises ValueError otherwise.
    """
    rank = max(len(a), len(b))
    padded_a = [1] * (rank - len(a)) + list(a)
    padded_b = [1] * (rank - len(b)) + list(b)
    shape = []
    for x, y in zip(padded_a, padded_b):
        if x != y and 1 not in (x, y):
            raise ValueError(f"cannot broadcast shapes {list(a)} and {list(b)}")
        shape.append(max(x, y))
    return shape


# -----------------------------
# Element-wise operations
# -----------------------------
def _verify_elementwise(op: IRDLOperation):
    """Requires each operand to be an integer or a tensor of rank >= 1, all
    of one element type. The result must be that integer type when every
    operand is a scalar, else a tensor of the operands' broadcast shape."""
    types = [v.type for v in op.operands]
    for t in types:
        if isinstance(t, TensorType) and len(t.shape) < 1:
            raise ValueError(f"{op.name}: a tensor operand must have rank >= 1, got {t}")

    elem_types = [t.element_type if isinstance(t, TensorType) else t for t in types]
    elem_t = elem_types[0]
    if any(t != elem_t for t in elem_types):
        raise ValueError(
            f"{op.name}: operands must share the same element type, got "
            + ", ".join(str(t) for t in types)
        )

    shape = []
    for t in types:
        shape = broadcast_shape(shape, shape_of(t))
    expected = TensorType(elem_t, shape) if shape else elem_t
    res_t = op.results[0].type
    if res_t != expected:
        raise ValueError(f"{op.name}: result type must be {expected}, got {res_t}")


@irdl_op_definition
class HCAdd(IRDLOperation):
    """lhs + rhs."""
    name = "hc.add"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


@irdl_op_definition
class HCMul(IRDLOperation):
    """lhs * rhs."""
    name = "hc.mul"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


@irdl_op_definition
class HCSub(IRDLOperation):
    """lhs - rhs."""
    name = "hc.sub"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


@irdl_op_definition
class HCRelu(IRDLOperation):
    """max(x, 0)."""
    name = "hc.relu"
    x = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


@irdl_op_definition
class HCPow(IRDLOperation):
    """base ** exp, defined for exp >= 0 (the lowering yields 1 for exp < 0)."""
    name = "hc.pow"
    base = operand_def(ScalarOrTensorInt)
    exp = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


@irdl_op_definition
class HCMax(IRDLOperation):
    """max(lhs, rhs)."""
    name = "hc.max"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


@irdl_op_definition
class HCMin(IRDLOperation):
    """min(lhs, rhs)."""
    name = "hc.min"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(ScalarOrTensorInt)

    def verify_(self):
        _verify_elementwise(self)


# -----------------------------
# Matmul
# -----------------------------
@irdl_op_definition
class HCMatmul(IRDLOperation):
    """(MxK) @ (KxN) -> (MxN) on the last two dims. Any leading batch dims
    are the same on lhs, rhs and res: (...xMxK) @ (...xKxN) -> (...xMxN)."""
    name = "hc.matmul"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        """Requires lhs/rhs/res to have one rank >= 2, share an element
        type and batch prefix (shape[:-2]), and satisfy MxK @ KxN -> MxN on
        shape[-2:]."""
        lhs_t, rhs_t, res_t = self.lhs.type, self.rhs.type, self.res.type

        for label, t in (("lhs", lhs_t), ("rhs", rhs_t), ("res", res_t)):
            if len(t.shape) < 2:
                raise ValueError(f"hc.matmul: {label} must have rank >= 2, got {t}")
        if not len(lhs_t.shape) == len(rhs_t.shape) == len(res_t.shape):
            raise ValueError(
                f"hc.matmul: lhs/rhs/res must all be the same rank, got "
                f"{len(lhs_t.shape)}/{len(rhs_t.shape)}/{len(res_t.shape)}"
            )

        if lhs_t.element_type != rhs_t.element_type or lhs_t.element_type != res_t.element_type:
            raise ValueError(
                f"hc.matmul: lhs/rhs/res must share the same element type, got "
                f"lhs={lhs_t.element_type}, rhs={rhs_t.element_type}, res={res_t.element_type}"
            )

        lhs_dims = [_dim(d) for d in lhs_t.shape]
        rhs_dims = [_dim(d) for d in rhs_t.shape]
        res_dims = [_dim(d) for d in res_t.shape]
        lhs_batch, rhs_batch, res_batch = lhs_dims[:-2], rhs_dims[:-2], res_dims[:-2]
        m, k = lhs_dims[-2:]
        k2, n = rhs_dims[-2:]
        rm, rn = res_dims[-2:]

        if lhs_batch != rhs_batch or lhs_batch != res_batch:
            raise ValueError(
                f"hc.matmul: batch dimensions must match across lhs/rhs/res, got "
                f"lhs={lhs_batch}, rhs={rhs_batch}, res={res_batch}"
            )
        if k != k2:
            raise ValueError(f"hc.matmul: inner dimensions must agree, got lhs K={k} vs rhs K={k2}")
        if (rm, rn) != (m, n):
            raise ValueError(
                f"hc.matmul: result shape must be ({m}x{n}), got ({rm}x{rn})"
            )


# -----------------------------
# Dialect: HiCompiler
# -----------------------------
HiCompiler = Dialect(
    "hc",
    (
        HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,   # element-wise ops
        HCMatmul,
    ),
    (),  # attrs
)

