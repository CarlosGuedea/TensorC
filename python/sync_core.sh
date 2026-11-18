#!/bin/sh
# Refresca el núcleo C vendorizado en python/tensorc/core/ desde la raíz
# del repositorio. Ejecutar tras modificar minitf.c, ndtensor.c, los
# headers públicos o el backend CUDA, antes de publicar el paquete Python.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/python/tensorc/core"
cp "$ROOT/minitf.c" "$ROOT/ndtensor.c" "$ROOT/minitf.h" "$PY/"
cp "$ROOT/tensorc_cuda.cu" "$ROOT/tensorc_cuda_internal.h" "$PY/"
cp "$ROOT/include/tensorc/ndtensor.h" "$ROOT/include/tensorc/tensorc.h" \
   "$PY/include/tensorc/"
echo "núcleo sincronizado en python/tensorc/core/"
