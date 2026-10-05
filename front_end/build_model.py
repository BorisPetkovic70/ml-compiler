"""Builders for the sample ONNX models (all INT32, opset 13).

Each builder checks the model with `onnx.checker` and saves it to `path`
(`_save`), by default `build/<name>.onnx`, creating the directory if needed. Running
this module writes all five models.
"""
import os
import onnx
from onnx import helper, TensorProto

# Base directory: build/ at the repository root
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "build")

MODEL_PATH = os.path.join(BASE_DIR, "score_model.onnx")
VEC_AFFINE_RELU_MODEL_PATH = os.path.join(BASE_DIR, "vec_affine_relu.onnx")
MATMUL_MODEL_PATH = os.path.join(BASE_DIR, "matmul.onnx")
CHAINED_TENSOR_MODEL_PATH = os.path.join(BASE_DIR, "chained_tensor_math.onnx")
BATCHED_MATMUL_MODEL_PATH = os.path.join(BASE_DIR, "batched_matmul.onnx")


def _save(model, path: str) -> None:
    """Checks `model` and saves it to `path`, creating its directory if
    needed (none for a bare file name)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved ONNX model to: {path}")


def build_score_model(path: str = MODEL_PATH):
    """Scalar model: score = relu(min(max(alpha * |x - mu| ** p, lo), hi)),
    with one scalar input `x`. Exercises Sub, Max, Pow, Mul, Min, Relu."""
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

    _save(model, path)




def build_vec_affine_relu_model(path: str = VEC_AFFINE_RELU_MODEL_PATH, n: int = 4):
    """Vector model: y = relu((a * x + b) * w), with vector inputs x, b of
    shape [n], a scalar input a, and a constant vector w that repeats
    [7, 2, 3, 5] to length n."""

    # ------------------------------------------------------------
    # Constant initializers
    # ------------------------------------------------------------
    w_init = helper.make_tensor("w", TensorProto.INT32, [n], ([7, 2, 3, 5] * n)[:n])

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

    _save(model, path)




def build_matmul_model(path: str = MATMUL_MODEL_PATH, m: int = 4, k: int = 4, n: int = 4):
    """Single-node model: C = A @ B, A:(MxK), B:(KxN), C:(MxN)."""

    a = helper.make_tensor_value_info("a", TensorProto.INT32, [m, k])
    b = helper.make_tensor_value_info("b", TensorProto.INT32, [k, n])
    c = helper.make_tensor_value_info("c", TensorProto.INT32, [m, n])

    nodes = [
        helper.make_node("MatMul", ["a", "b"], ["c"], name="matmul"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="MatmulGraph",
        inputs=[a, b],
        outputs=[c],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="matmul_builder",
    )

    _save(model, path)




def build_chained_tensor_model(path: str = CHAINED_TENSOR_MODEL_PATH, m: int = 4, k: int = 4, n: int = 4):
    """Tensor chain: Y = relu((A @ B) + C), A:(MxK), B:(KxN), C and Y:(MxN).
    Its two intermediate tensors become temporary buffers after
    bufferization."""

    # ------------------------------------------------------------
    # Inputs / Output (Strictly rank-2 to avoid broadcasting)
    # ------------------------------------------------------------
    a = helper.make_tensor_value_info("a", TensorProto.INT32, [m, k])
    b = helper.make_tensor_value_info("b", TensorProto.INT32, [k, n])
    c = helper.make_tensor_value_info("c", TensorProto.INT32, [m, n])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [m, n])

    # ------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------
    nodes = [
        # mm_res = A @ B
        helper.make_node("MatMul", ["a", "b"], ["mm_res"], name="matmul_ab"),

        # add_res = mm_res + C
        helper.make_node("Add", ["mm_res", "c"], ["add_res"], name="add_c"),

        # y = ReLU(add_res)
        helper.make_node("Relu", ["add_res"], ["y"], name="relu_out"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="ChainedTensorGraph",
        inputs=[a, b, c],
        outputs=[y],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="chained_tensor_builder",
    )

    _save(model, path)


def build_batched_matmul_model(
    path: str = BATCHED_MATMUL_MODEL_PATH, batch: int = 2, m: int = 4, k: int = 4, n: int = 4
):
    """Single-node batched model: C = A @ B, A:(BxMxK), B:(BxKxN), C:(BxMxN)."""

    a =helper.make_tensor_value_info("a", TensorProto.INT32, [batch, m, k])
    b = helper.make_tensor_value_info("b", TensorProto.INT32, [batch, k, n])
    c = helper.make_tensor_value_info("c", TensorProto.INT32, [batch, m, n])

    nodes = [
        helper.make_node("MatMul", ["a", "b"], ["c"], name="batched_matmul"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="BatchedMatmulGraph",
        inputs=[a, b],
        outputs=[c],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="batched_matmul_builder",
    )

    _save(model, path)


if __name__ == "__main__":
    build_score_model()
    build_vec_affine_relu_model()
    build_matmul_model()
    build_chained_tensor_model()
    build_batched_matmul_model()
