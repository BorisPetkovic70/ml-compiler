"""The `hc` dialect: ONNX-shaped integer ops, one family each for scalars,
vectors (`*_vec`), and tensors (`*_tensor`, `hc.matmul`). See docs/DESIGN.md
Sections 1-2.

`VecInt`/`TensorInt` constrain only the element type. Every rule relating the
types or shapes of an op's operands and result is enforced by that op's
`verify_()`, which raises `ValueError`. Tensor ops accept rank 2, or rank 3
with a leading batch dim.

An op belongs to the dialect only once it is listed in the `HiCompiler` tuple
at the bottom of this file (checklist: docs/HOW_TO_ADD_AN_OP.md).
"""
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.dialects.builtin import IntegerType, VectorType, TensorType

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
# Vector operations
# -----------------------------
# "Vector of integers" type constraint (any rank/shape, integer element type)
VecInt = VectorType.constr(element_type=IntegerType)


def _verify_bin_same_vec_type(op: IRDLOperation):
    """Requires lhs, rhs, and res to have the identical vector type."""
    lhs_t = op.lhs.type
    rhs_t = op.rhs.type
    res_t = op.res.type
    if lhs_t != rhs_t:
        raise ValueError(
            f"{op.name}: lhs and rhs must have the same vector type, got {lhs_t} vs {rhs_t}"
        )
    if res_t != lhs_t:
        raise ValueError(
            f"{op.name}: result must match operand vector type, got res={res_t}, operand={lhs_t}"
        )


@irdl_op_definition
class HCAddVec(IRDLOperation):
    name = "hc.add_vec"
    lhs = operand_def(VecInt)
    rhs = operand_def(VecInt)
    res = result_def(VecInt)

    def verify_(self):
        _verify_bin_same_vec_type(self)


@irdl_op_definition
class HCSubVec(IRDLOperation):
    name = "hc.sub_vec"
    lhs = operand_def(VecInt)
    rhs = operand_def(VecInt)
    res = result_def(VecInt)

    def verify_(self):
        _verify_bin_same_vec_type(self)


@irdl_op_definition
class HCMulVec(IRDLOperation):
    """scalar * vector -> vector: multiplies every lane by `scalar`."""
    name = "hc.mul_vec"
    scalar = operand_def(IntegerType)
    vec = operand_def(VecInt)
    res = result_def(VecInt)

    def verify_(self):
        if self.res.type != self.vec.type:
            raise ValueError(
                f"{self.name}: result must match vector operand type, "
                f"got res={self.res.type}, vec={self.vec.type}"
            )


@irdl_op_definition
class HCReluVec(IRDLOperation):
    """Element-wise max(x, 0) over a vector."""
    name = "hc.relu_vec"
    x = operand_def(VecInt)
    res = result_def(VecInt)

    def verify_(self):
        if self.res.type != self.x.type:
            raise ValueError(
                f"{self.name}: result must match operand vector type, "
                f"got res={self.res.type}, x={self.x.type}"
            )

@irdl_op_definition
class HCMulVecVec(IRDLOperation):
    """Element-wise vector * vector -> vector."""
    name = "hc.mul_vec_vec"
    lhs = operand_def(VecInt)
    rhs = operand_def(VecInt)
    res = result_def(VecInt)

    def verify_(self):
            _verify_bin_same_vec_type(self)


# -----------------------------
# Tensor operations
# -----------------------------
# "Tensor of integers" type constraint (any element width; rank checked in verify_)
TensorInt = TensorType.constr(element_type=IntegerType)


def _dim(int_attr) -> int:
    return int_attr.data


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


def _verify_bin_same_tensor_type(op: IRDLOperation):
    """Requires lhs, rhs, and res to have the identical rank-2 or rank-3
    tensor type."""
    lhs_t, rhs_t, res_t = op.lhs.type, op.rhs.type, op.res.type
    if len(lhs_t.shape) not in (2, 3):
        raise ValueError(
            f"{op.name}: operands must be rank-2, or rank-3 with a leading batch "
            f"dim, got {lhs_t}"
        )
    if lhs_t != rhs_t:
        raise ValueError(
            f"{op.name}: lhs and rhs must have the same tensor type, got {lhs_t} vs {rhs_t}"
        )
    if res_t != lhs_t:
        raise ValueError(
            f"{op.name}: result must match operand tensor type, got res={res_t}, operand={lhs_t}"
        )


@irdl_op_definition
class HCAddTensor(IRDLOperation):
    """Element-wise tensor + tensor -> tensor."""
    name = "hc.add_tensor"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_bin_same_tensor_type(self)


@irdl_op_definition
class HCSubTensor(IRDLOperation):
    """Element-wise tensor - tensor -> tensor."""
    name = "hc.sub_tensor"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_bin_same_tensor_type(self)


@irdl_op_definition
class HCMulTensor(IRDLOperation):
    """Element-wise tensor * tensor -> tensor. There is no scalar * tensor
    variant."""
    name = "hc.mul_tensor"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_bin_same_tensor_type(self)


@irdl_op_definition
class HCReluTensor(IRDLOperation):
    """Element-wise max(x, 0) over a tensor."""
    name = "hc.relu_tensor"
    x = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        if len(self.x.type.shape) not in (2, 3):
            raise ValueError(
                f"hc.relu_tensor: operand must be rank-2, or rank-3 with a "
                f"leading batch dim, got {self.x.type}"
            )
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
        HCAddVec, HCSubVec, HCMulVec, HCReluVec, HCMulVecVec,  # vector ops
        HCMatmul, HCAddTensor, HCSubTensor, HCMulTensor, HCReluTensor,  # tensor ops
    ),
    (),  # attrs
)

