#!/usr/bin/env bash
# Compiles a model, runs it, and checks the executable against the interpreter.
# Usage: ./full_compiler.sh [--interchange] [--tile T] [--vectorize W] [model.onnx] [args...]

set -e   # stop at the first failing stage

# The leading options choose the loop passes. Both Python drivers get them, so
# the interpreter and the executable run the same loops.
FLAGS=()
while [[ "$1" == --* ]]; do
    case "$1" in
        --tile|--vectorize) FLAGS+=("$1" "$2"); shift 2 ;;
        *) FLAGS+=("$1"); shift ;;
    esac
done

MODEL_NAME=${1:-score_model.onnx}
MODEL_STEM="$(basename "${MODEL_NAME%.onnx}")"

echo "Using model: $MODEL_NAME"

python3 hc_main.py "${FLAGS[@]}" "$MODEL_NAME"
expected="$(python3 hc_interpret.py "${FLAGS[@]}" "$MODEL_NAME" "${@:2}")"
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
