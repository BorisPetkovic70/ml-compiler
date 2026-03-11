import onnx
from onnx import helper, TensorProto

MODEL_PATH = "score_model.onnx"
def build_score_model(path: str = MODEL_PATH):
    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    MU = 40.0
    ALPHA = 2.0
    P = 2.0
    LO = 0.0
    HI = 1000.0

    # ------------------------------------------------------------
    # Input / Output
    # ------------------------------------------------------------
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, [])

    # ------------------------------------------------------------
    # Constant initializers
    # ------------------------------------------------------------
    mu_init = helper.make_tensor("mu", TensorProto.FLOAT, [], [MU])
    a_init  = helper.make_tensor("alpha", TensorProto.FLOAT, [], [ALPHA])
    p_init  = helper.make_tensor("p", TensorProto.FLOAT, [], [P])
    lo_init = helper.make_tensor("lo", TensorProto.FLOAT, [], [LO])
    hi_init = helper.make_tensor("hi", TensorProto.FLOAT, [], [HI])

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


if __name__ == "__main__":
    build_score_model()
