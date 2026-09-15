from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.dialects.builtin import IntegerType, VectorType

# -----------------------------
# Operations
# -----------------------------
@irdl_op_definition
class HCAdd(IRDLOperation):
    name = "hc.add"

    # Two i32 operands
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)

    # One i32 result
    res = result_def(IntegerType)

@irdl_op_definition
class HCMul(IRDLOperation):
    name = "hc.mul"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)


@irdl_op_definition
class HCSub(IRDLOperation):
    name = "hc.sub"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

@irdl_op_definition
class HCRelu(IRDLOperation):
    name = "hc.relu"
    x = operand_def(IntegerType)
    res = result_def(IntegerType)

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
# Dialect: HiCompiler
# -----------------------------
HiCompiler = Dialect(
    "hc",
    (
        HCAdd, HCMul, HCSub, HCRelu, HCPow, HCMax, HCMin,   # scalar ops
        HCAddVec, HCSubVec, HCMulVec, HCReluVec, HCMulVecVec         # vector ops
    ),
    (),  # attrs
)

