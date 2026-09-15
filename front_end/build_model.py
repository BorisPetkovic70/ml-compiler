import onnx
from onnx import helper, TensorProto

MODEL_PATH = "score_model.onnx"
def build_score_model(path: str = MODEL_PATH):
    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    MU = 40
    ALPHA = 2
    P = 2
    LO = 0
    HI = 1000

    # ------------------------------------------------------------
    # Input / Output
    # ------------------------------------------------------------
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [])
    y = helper.make_tensor_value_info("score", TensorProto.INT32, [])

    # ------------------------------------------------------------
    # Constant initializers
    # ------------------------------------------------------------
    mu_init = helper.make_tensor("mu", TensorProto.INT32, [], [MU])
    a_init  = helper.make_tensor("alpha", TensorProto.INT32, [], [ALPHA])
    p_init  = helper.make_tensor("p", TensorProto.INT32, [], [P])
    lo_init = helper.make_tensor("lo", TensorProto.INT32, [], [LO])
    hi_init = helper.make_tensor("hi", TensorProto.INT32, [], [HI])

    # ------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------
    nodes = [
        # d1 = x - mu
        helper.make_node("Sub", ["x", "mu"], ["d1"], name="sub_x_mu"),

        # d2 = mu - x
        helper.make_node("Sub", ["mu", "x"], ["d2"], name="sub_mu_x"),

        # abs_d = max(d1, d2)
        helper.make_node("Max", ["d1", "d2"], ["abs_d"], name="abs_max"),

        # e = pow(abs_d, p)
        helper.make_node("Pow", ["abs_d", "p"], ["e"], name="pow"),

        # s = alpha * e
        helper.make_node("Mul", ["alpha", "e"], ["s"], name="scale"),

        # lower = max(s, lo)
        helper.make_node("Max", ["s", "lo"], ["lower"], name="clamp_low"),

        # clamped = min(lower, hi)
        helper.make_node("Min", ["lower", "hi"], ["clamped"], name="clamp_high"),

        # score = relu(clamped)
        helper.make_node("Relu", ["clamped"], ["score"], name="relu"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="ScoreGraph",
        inputs=[x],
        outputs=[y],
        initializer=[mu_init, a_init, p_init, lo_init, hi_init],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="score_builder",
    )

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved ONNX model to: {path}")


VEC_AFFINE_RELU_MODEL_PATH = "vec_affine_relu.onnx"

def build_vec_affine_relu_model(path: str = VEC_AFFINE_RELU_MODEL_PATH, n: int = 4):
    """Build a simple 3-node vector model: y = relu(a * x + b), shape [n].

    All tensors are vector-shaped [n]; no broadcasting involved.
    """


    # ------------------------------------------------------------
    # Constant initializers
    # ------------------------------------------------------------
    w_init = helper.make_tensor("w", TensorProto.INT32, [n], [7,2,3,5])

    # ------------------------------------------------------------
    # Inputs / Output  (all vector-shaped: [n])
    # ------------------------------------------------------------
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [n])
    a = helper.make_tensor_value_info("a", TensorProto.INT32, [])
    b = helper.make_tensor_value_info("b", TensorProto.INT32, [n])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [n])

    # ------------------------------------------------------------
    # Graph nodes (3 total)
    # ------------------------------------------------------------
    nodes = [
        # s = a * x
        helper.make_node("Mul", ["a", "x"], ["s"], name="mul_ax"),

        # t = s + b
        helper.make_node("Add", ["s", "b"], ["t"], name="add_b"),

        #p = t*w (vec mul)
        helper.make_node("Mul", ["t","w"], ["p"], name="mul_tw"),

        # y = relu(p)
        helper.make_node("Relu", ["p"], ["y"], name="relu"),


    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="VecAffineReluGraph",
        inputs=[x, a, b],
        outputs=[y],
        initializer=[w_init]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="vec_affine_relu_builder",
    )

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved ONNX model to: {path}")


if __name__ == "__main__":
    build_score_model()
    build_vec_affine_relu_model()
