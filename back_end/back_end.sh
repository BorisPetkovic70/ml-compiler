#!/usr/bin/env bash

set -e   # stop at the first failing tool instead of running on with stale files

MODEL_NAME="$1"
# basename: strip the suffix, then any directory component -- "build/matmul.onnx"
# must give the same stem "matmul" that hc_main.py's Path(...).stem computes, or
# BUILD_DIR below gets appended on top of the directory already in the argument
# (".../build/build/...").
MODEL_STEM="$(basename "${MODEL_NAME%.onnx}")"
echo "Compiling backend..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#TOOLCHAIN_BIN_DIR="$SCRIPT_DIR/../../LLVM_IR/llvm-project/build/bin"
TOOLCHAIN_BIN_DIR="$SCRIPT_DIR/../../../../MLIR_clone/llvm-project/build/bin"

BUILD_DIR=${SCRIPT_DIR}/../build

# memref-bearing models (anything that went through bufferization) need the memref
# passes plus a C-ABI wrapper; they are harmless no-ops for scalar/vector models.
#   --llvm-request-c-wrappers  emits _mlir_ciface_<fn>, which takes/returns memrefs as
#                              pointers to descriptor structs the C harness can build
#   --expand-strided-metadata  lowers memref metadata ops the memref pass expects
#   --finalize-memref-to-llvm  memref.alloc/load/store -> llvm (malloc + getelementptr)
#   --lower-affine             any affine.* produced by later loop passes
${TOOLCHAIN_BIN_DIR}/mlir-opt ${BUILD_DIR}/${MODEL_STEM}_lowered.mlir \
  --llvm-request-c-wrappers \
  --expand-strided-metadata \
  --finalize-memref-to-llvm \
  --lower-affine \
  -convert-vector-to-llvm \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ${BUILD_DIR}/${MODEL_STEM}_llvm.mlir
echo "Created file: ${BUILD_DIR}/${MODEL_STEM}_llvm.mlir"

${TOOLCHAIN_BIN_DIR}/mlir-translate ${BUILD_DIR}/${MODEL_STEM}_llvm.mlir \
    -mlir-to-llvmir -o ${BUILD_DIR}/${MODEL_STEM}_out.ll
echo "Created file: ${BUILD_DIR}/${MODEL_STEM}_out.ll"

${TOOLCHAIN_BIN_DIR}/llc -O2 -filetype=asm \
    ${BUILD_DIR}/${MODEL_STEM}_out.ll -o ${BUILD_DIR}/${MODEL_STEM}_out.s
echo "Created file: ${BUILD_DIR}/${MODEL_STEM}_out.s"

${TOOLCHAIN_BIN_DIR}/llc -filetype=obj ${BUILD_DIR}/${MODEL_STEM}_out.ll \
    -o ${BUILD_DIR}/${MODEL_STEM}_out.o
echo "Created file: ${BUILD_DIR}/${MODEL_STEM}_out.o"

clang ${SCRIPT_DIR}/${MODEL_STEM}_harness.c ${BUILD_DIR}/${MODEL_STEM}_out.o -o \
    ${BUILD_DIR}/${MODEL_STEM}_run

echo "Created executable: ${BUILD_DIR}/${MODEL_STEM}_run"
echo "Running executable..."

#${BUILD_DIR}/${MODEL_STEM}_run 1 1 1 1 100 20 -200 -500 70
${BUILD_DIR}/${MODEL_STEM}_run 47 12 89 7 56 74 21 98 15 63 8 41 77 29 52 86 11 67 34 95 4 58 72 19 83 46 1 90 27 61 38 54
