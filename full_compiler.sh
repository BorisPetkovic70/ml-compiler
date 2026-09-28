#!/usr/bin/env bash

MODEL_NAME=${1:-score_model.onnx}

echo "Using model: $MODEL_NAME"

python3 hc_main.py "$MODEL_NAME"
back_end/back_end.sh "$MODEL_NAME" "${@:2}"
