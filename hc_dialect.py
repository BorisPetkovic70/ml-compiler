"""The `hc` dialect: ONNX-shaped integer ops, one family for scalars and one
for tensors (`*_tensor`, `hc.matmul`). See docs/DESIGN.md Sections 1-2.

`TensorInt` constrains only the element type. Every rule relating the types
or shapes of an op's operands and result is enforced by that op's
`verify_()`, which raises `ValueError`. Element-wise tensor ops accept any
rank >= 1; `hc.matmul` accepts rank 2, or rank 3 with a leading batch dim.
The binary element-wise tensor ops broadcast their operands with NumPy's
rules (`broadcast_shape`), and either operand may be a scalar.

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
# Operations
# -----------------------------


def _verify_bin_same_int_type(op: IRDLOperation):
    """Requires lhs, rhs, and res to have the identical integer type."""
    if op.lhs.type != op.rhs.type or op.res.type != op.lhs.type:
        raise ValueError(
            f"{op.name}: operands and result must have the exact same integer type, "
            f"got lhs={op.lhs.type}, rhs={op.rhs.type}, res={op.res.type}"
        )

@irdl_op_definition
class HCAdd(IRDLOperation):
    name = "hc.add"

    # Two i32 operands
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)

    # One i32 result
    res = result_def(IntegerType)

    def verify_(self):
        _verify_bin_same_int_type(self)

@irdl_op_definition
class HCMul(IRDLOperation):
    name = "hc.mul"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

    def verify_(self):
        _verify_bin_same_int_type(self)


@irdl_op_definition
class HCSub(IRDLOperation):
    name = "hc.sub"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

    def verify_(self):
        _verify_bin_same_int_type(self)

@irdl_op_definition
class HCRelu(IRDLOperation):
    name = "hc.relu"
    x = operand_def(IntegerType)
    res = result_def(IntegerType)

    def verify_(self):
        if self.res.type != self.x.type:
            raise ValueError(f"hc.relu: result type must match operand type")

@irdl_op_definition
class HCPow(IRDLOperation):
    """base ** exp, defined for exp >= 0 (the lowering yields 1 for exp < 0)."""
    name = "hc.pow"
    base = operand_def(IntegerType)
    exp  = operand_def(IntegerType)
    res  = result_def(IntegerType)

    def verify_(self):
        if self.base.type != self.exp.type or self.res.type != self.base.type:
            raise ValueError("hc.pow: base, exp, and res must have same integer type")

@irdl_op_definition
class HCMax(IRDLOperation):
    name = "hc.max"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

    def verify_(self):
        if self.lhs.type != self.rhs.type or self.res.type != self.lhs.type:
            raise ValueError("hc.max: operands and result must have the same integer type")

@irdl_op_definition
class HCMin(IRDLOperation):
    name = "hc.min"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

    def verify_(self):
        if self.lhs.type != self.rhs.type or self.res.type != self.lhs.type:
            raise ValueError("hc.min: operands and result must have the same integer type")

# -----------------------------
# Tensor operations
# -----------------------------
# "Tensor of integers" type constraint (any element width; rank checked in verify_)
TensorInt = TensorType.constr(element_type=IntegerType)


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


@irdl_op_definition
class HCMatmul(IRDLOperation):
    """(MxK) @ (KxN) -> (MxN), or batched (BxMxK) @ (BxKxN) -> (BxMxN)."""
    name = "hc.matmul"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        """Requires lhs/rhs/res to be all rank-2 or all rank-3, share an
        element type and batch prefix (shape[:-2]), and satisfy
        MxK @ KxN -> MxN on shape[-2:]."""
        lhs_t, rhs_t, res_t = self.lhs.type, self.rhs.type, self.res.type

        for label, t in (("lhs", lhs_t), ("rhs", rhs_t), ("res", res_t)):
            if len(t.shape) not in (2, 3):
                raise ValueError(
                    f"hc.matmul: {label} must be rank-2, or rank-3 with a leading "
                    f"batch dim, got {t}"
                )
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


# An operand of a broadcasting op: a scalar, or a tensor of integers.
ScalarOrTensorInt = base(IntegerType) | TensorInt


def _verify_broadcast_binop(op: IRDLOperation):
    """Requires each operand to be an integer or a tensor of rank >= 1, at
    least one of them a tensor, and res to be a tensor of the operands'
    broadcast shape. Operands and result share one element type."""
    lhs_t, rhs_t, res_t = op.lhs.type, op.rhs.type, op.res.type
    tensors = [t for t in (lhs_t, rhs_t) if isinstance(t, TensorType)]
    if not tensors:
        raise ValueError(
            f"{op.name}: at least one operand must be a tensor, got {lhs_t} and {rhs_t}"
        )
    for t in tensors:
        if len(t.shape) < 1:
            raise ValueError(f"{op.name}: operands must have rank >= 1, got {t}")
    for t in (lhs_t, rhs_t):
        elem_t = t.element_type if isinstance(t, TensorType) else t
        if elem_t != res_t.element_type:
            raise ValueError(
                f"{op.name}: operands and result must share the same element type, "
                f"got lhs={lhs_t}, rhs={rhs_t}, res={res_t}"
            )
    shape = broadcast_shape(shape_of(lhs_t), shape_of(rhs_t))
    if shape_of(res_t) != shape:
        raise ValueError(f"{op.name}: result shape must be {shape}, got {res_t}")


@irdl_op_definition
class HCAddTensor(IRDLOperation):
    """Element-wise lhs + rhs -> tensor, with broadcasting."""
    name = "hc.add_tensor"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_broadcast_binop(self)


@irdl_op_definition
class HCSubTensor(IRDLOperation):
    """Element-wise lhs - rhs -> tensor, with broadcasting."""
    name = "hc.sub_tensor"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_broadcast_binop(self)


@irdl_op_definition
class HCMulTensor(IRDLOperation):
    """Element-wise lhs * rhs -> tensor, with broadcasting."""
    name = "hc.mul_tensor"
    lhs = operand_def(ScalarOrTensorInt)
    rhs = operand_def(ScalarOrTensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_broadcast_binop(self)


@irdl_op_definition
class HCReluTensor(IRDLOperation):
    """Element-wise max(x, 0) over a tensor."""
    name = "hc.relu_tensor"
    x = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        if len(self.x.type.shape) < 1:
            raise ValueError(f"hc.relu_tensor: operand must have rank >= 1, got {self.x.type}")
        if self.res.type != self.x.type:
            raise ValueError(
                f"hc.relu_tensor: result must match operand tensor type, "
                f"got res={self.res.type}, x={self.x.type}"
            )


# -----------------------------
# Dialect: HiCompiler
# -----------------------------
HiCompiler = Dialect(
    "hc",
    (
        HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,   # scalar ops
        HCMatmul, HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor,  # tensor ops
    ),
    (),  # attrs
)

