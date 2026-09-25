"""The hc dialect: ONNX-shaped operations over scalars, vectors, and tensors.

Sits above arith./scf./vector./tensor. specifically so "what the model computes"
(this file) and "how that computation becomes loops and memory" (middle_end/) are
two separate, independently-testable decisions -- see docs/DESIGN.md Section 1.

VecInt/TensorInt (below) constrain element type only -- neither binds shapes
across an op's operands. This is why every op with a shape relationship between
its operands (mismatched-width scalar ops, same-shape vector ops, hc.matmul's
MxK*KxN->MxN, the tensor elementwise family's exact-shape match) needs its own
hand-written verify_(): the type system alone cannot express those constraints.
See docs/DESIGN.md Section 2 for why scalars, vectors, and tensors are modeled
as genuinely different op families rather than one polymorphic op per operation.

Every op defined here must also be added to the HiCompiler Dialect tuple at the
bottom of this file -- an op that isn't registered there isn't part of the
dialect at all, however correctly it's otherwise defined. See
docs/HOW_TO_ADD_AN_OP.md for the full checklist.
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
    """scalar * vector -> vector (element-wise scale)"""
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
    """ReLU over a vector: max(x, 0) element-wise"""
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
    """vector * vector -> vector (element-wise)"""
    name = "hc.mul_vec_vec"
    lhs = operand_def(VecInt)
    rhs = operand_def(VecInt)
    res = result_def(VecInt)

    def verify_(self):
            _verify_bin_same_vec_type(self)


# -----------------------------
# Tensor operations
# -----------------------------
# "2D tensor of integers" type constraint (any element width, rank checked in verify_)
TensorInt = TensorType.constr(element_type=IntegerType)


def _dim(int_attr) -> int:
    return int_attr.data


@irdl_op_definition
class HCMatmul(IRDLOperation):
    """2D matrix multiply: (MxK) * (KxN) -> (MxN)"""
    name = "hc.matmul"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        lhs_t, rhs_t, res_t = self.lhs.type, self.rhs.type, self.res.type

        for label, t in (("lhs", lhs_t), ("rhs", rhs_t), ("res", res_t)):
            if len(t.shape) != 2:
                raise ValueError(f"hc.matmul: {label} must be a rank-2 tensor, got {t}")

        if lhs_t.element_type != rhs_t.element_type or lhs_t.element_type != res_t.element_type:
            raise ValueError(
                f"hc.matmul: lhs/rhs/res must share the same element type, got "
                f"lhs={lhs_t.element_type}, rhs={rhs_t.element_type}, res={res_t.element_type}"
            )

        m, k = _dim(list(lhs_t.shape)[0]), _dim(list(lhs_t.shape)[1])
        k2, n = _dim(list(rhs_t.shape)[0]), _dim(list(rhs_t.shape)[1])
        rm, rn = _dim(list(res_t.shape)[0]), _dim(list(res_t.shape)[1])

        if k != k2:
            raise ValueError(f"hc.matmul: inner dimensions must agree, got lhs K={k} vs rhs K={k2}")
        if (rm, rn) != (m, n):
            raise ValueError(
                f"hc.matmul: result shape must be ({m}x{n}), got ({rm}x{rn})"
            )


def _verify_bin_same_tensor_type(op: IRDLOperation):
    """Shared verify_() for tensor⊙tensor elementwise ops: unlike hc.matmul (whose
    operand shapes legitimately differ), lhs/rhs/res must be the exact same rank-2
    tensor type -- TensorInt alone doesn't bind shape across operands (see module
    docstring), so this still needs a hand-written check."""
    lhs_t, rhs_t, res_t = op.lhs.type, op.rhs.type, op.res.type
    if len(lhs_t.shape) != 2:
        raise ValueError(f"{op.name}: operands must be rank-2 tensors, got {lhs_t}")
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
    """Element-wise tensor + tensor -> tensor (rank-2, exact same shape)"""
    name = "hc.add_tensor"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_bin_same_tensor_type(self)


@irdl_op_definition
class HCSubTensor(IRDLOperation):
    """Element-wise tensor - tensor -> tensor (rank-2, exact same shape)"""
    name = "hc.sub_tensor"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_bin_same_tensor_type(self)


@irdl_op_definition
class HCMulTensor(IRDLOperation):
    """Element-wise tensor * tensor -> tensor (rank-2, exact same shape).
    Deliberately no scalar*tensor variant (unlike hc.mul_vec) -- not asked for,
    and would need a tensor-analogue of vector.broadcast to lower."""
    name = "hc.mul_tensor"
    lhs = operand_def(TensorInt)
    rhs = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        _verify_bin_same_tensor_type(self)


@irdl_op_definition
class HCReluTensor(IRDLOperation):
    """ReLU over a rank-2 tensor: max(x, 0) element-wise"""
    name = "hc.relu_tensor"
    x = operand_def(TensorInt)
    res = result_def(TensorInt)

    def verify_(self):
        if len(self.x.type.shape) != 2:
            raise ValueError(f"hc.relu_tensor: operand must be a rank-2 tensor, got {self.x.type}")
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

