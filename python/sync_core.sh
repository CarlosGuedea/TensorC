#!/bin/sh
# Refresca el núcleo C vendorizado en python/tensorc/core/ desde la raíz
# del repositorio. Ejecutar tras modificar minitf.c, ndtensor.c, los
# headers públicos o cualquier backend, antes de publicar el paquete Python.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/python/tensorc/core"
cp "$ROOT/minitf.c" "$ROOT/ndtensor.c" "$ROOT/rl.c" "$ROOT/minitf.h" "$PY/"
cp "$ROOT/tensorc_cuda.cu" "$ROOT/tensorc_device_internal.h" "$ROOT/tensorc_metal.mm" "$PY/"
cp "$ROOT/include/tensorc/ndtensor.h" "$ROOT/include/tensorc/tensorc.h" \
   "$ROOT/include/tensorc/rl.h" \
   "$PY/include/tensorc/"
echo "núcleo sincronizado en python/tensorc/core/"
