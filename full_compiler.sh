#!/usr/bin/env bash
# Compiles a model, runs it, and checks the executable against the interpreter.
# Usage: ./full_compiler.sh [model.onnx] [args...]

set -e   # stop at the first failing stage

MODEL_NAME=${1:-score_model.onnx}
MODEL_STEM="$(basename "${MODEL_NAME%.onnx}")"

echo "Using model: $MODEL_NAME"

python3 hc_main.py "$MODEL_NAME"
expected="$(python3 hc_interpret.py "$MODEL_NAME" "${@:2}")"
back_end/back_end.sh "$MODEL_NAME" "${@:2}"

# back_end.sh's own output is mixed with the executable's, so run the
# executable again to capture its result alone.
actual="$(build/${MODEL_STEM}_run "${@:2}")"

if [ "$expected" == "$actual" ]; then
    echo "OK: interpreter and executable agree"
else
    echo "MISMATCH: interpreter and executable disagree"
    echo "interpreter:"
    echo "$expected"
    echo "executable:"
    echo "$actual"
    exit 1
fi
