#!/usr/bin/env bash
# Compiles and runs every sample model through full_compiler.sh with fixed inputs.
# Usage: [TOOLCHAIN_BIN_DIR=/path/to/llvm/bin] ./run_all_models.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

IDENTITY_4X4="1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1"
DOUBLE_4X4="2 0 0 0  0 2 0 0  0 0 2 0  0 0 0 2"

# Rebuild the models so their shapes match the inputs below (full_compiler.sh
# reuses whatever build/<model>.onnx already exists).
python3 front_end/build_model.py

run() {
    echo
    echo "==================== $1 ===================="
    ./full_compiler.sh "$@"
}

# x = 25: relu(min(max(2 * |25 - 40|^2, 0), 1000)) = 450
run score_model.onnx 25

# x = [1 2 3 4], a = 3, b = [-100 -5 0 2]: relu((a*x + b) * [7 2 3 5]) = [0, 2, 27, 70]
run vec_affine_relu.onnx 1 2 3 4  3  -100 -5 0 2

# A = 1..16, B = identity: A @ B = A
run matmul.onnx $(seq 1 16) $IDENTITY_4X4

# A = 1..16, B = identity, C = -8 everywhere: relu(A - 8), so the first two rows are 0
run chained_tensor_math.onnx $(seq 1 16) $IDENTITY_4X4 \
    -8 -8 -8 -8  -8 -8 -8 -8  -8 -8 -8 -8  -8 -8 -8 -8

# A = 1..32 (two 4x4 batches), B = [identity, 2 * identity]: [A0, 2 * A1]
run batched_matmul.onnx $(seq 1 32) $IDENTITY_4X4 $DOUBLE_4X4
