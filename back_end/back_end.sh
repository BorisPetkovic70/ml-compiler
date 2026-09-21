#/bin/sh
MODEL_NAME="$1"
MODEL_STEM="${MODEL_NAME%.onnx}"
echo "Compiling backend..."

# More robust way to determine script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN_BIN_DIR="$SCRIPT_DIR/../../LLVM_IR/llvm-project/build/bin"

BUILD_DIR=${SCRIPT_DIR}/../build

${TOOLCHAIN_BIN_DIR}/mlir-opt ${BUILD_DIR}/${MODEL_STEM}_lowered.mlir \
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

${BUILD_DIR}/${MODEL_STEM}_run 1 1 1 1 100 20 -200 -500 70
