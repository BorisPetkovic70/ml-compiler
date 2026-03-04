from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.dialects.builtin import IntegerType

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

# -----------------------------
# Dialect: HiCompiler
# -----------------------------
HiCompiler = Dialect(
    "hc",
    (HCAdd, HCMul, HCSub, HCRelu, HCPow),      # register operations here
    (),  # attrs
)

