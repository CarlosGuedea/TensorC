#!/bin/bash
# Reconstruye TensorC con CUDA en WSL y corre la suite en ambos modos.
# Se invoca desde Windows: wsl.exe -d Ubuntu -- bash build_wsl.sh
set -e
export PATH="$HOME/tensorc-env/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/tensorc-env/lib:$LD_LIBRARY_PATH"
SRC="/mnt/c/Users/guede/OneDrive/Documents/Proyectos/IA/TensorC-CUDA"

cp "$SRC/ndtensor.c" "$SRC/tensorc_cuda.cu" "$SRC/tensorc_cuda_internal.h" \
   "$SRC/minitf.c" "$SRC/minitf.h" "$SRC/CMakeLists.txt" ~/tensorc-src/
mkdir -p ~/tensorc-src/include/tensorc ~/tensorc-src/tests
cp "$SRC/include/tensorc/ndtensor.h" "$SRC/include/tensorc/tensorc.h" \
   ~/tensorc-src/include/tensorc/
cp "$SRC/tests/"*.c ~/tensorc-src/tests/
cp "$SRC/python/tensorc/_tensorc.c" ~/tensorc-src/python/tensorc/
cp "$SRC/python/tensorc/core/ndtensor.c" \
   "$SRC/python/tensorc/core/tensorc_cuda.cu" \
   "$SRC/python/tensorc/core/tensorc_cuda_internal.h" \
   "$SRC/python/tensorc/core/minitf.c" "$SRC/python/tensorc/core/minitf.h" \
   ~/tensorc-src/python/tensorc/core/
cp "$SRC/python/tensorc/core/include/tensorc/ndtensor.h" \
   "$SRC/python/tensorc/core/include/tensorc/tensorc.h" \
   ~/tensorc-src/python/tensorc/core/include/tensorc/

cd ~/tensorc-src
echo "=== BUILD ==="
cmake --build build-cuda -j 2>&1 | grep -E "error|warning: #|tensorc_cuda|ndtensor" | tail -8 || true
echo "=== PYEXT ==="
cd ~/tensorc-src/python
TENSORC_CUDA=1 ~/tensorc-env/bin/python setup.py build_ext --inplace 2>&1 | tail -3
cd ~/tensorc-src
echo "=== CTEST normal ==="
ctest --test-dir build-cuda --output-on-failure 2>&1 | tail -4
echo "=== CTEST TC_CUDA_MIN_ELEMS=0 ==="
TC_CUDA_MIN_ELEMS=0 ctest --test-dir build-cuda --output-on-failure 2>&1 | tail -4
