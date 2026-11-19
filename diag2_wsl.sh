#!/bin/bash
# Separa el coste host-side (Python+binding+grafo) del coste de lanzamiento
# GPU: misma cadena de ops con tensores diminutos (camino CPU, sin CUDA).
set -e
export PATH="$HOME/tensorc-env/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/tensorc-env/lib:$LD_LIBRARY_PATH"
cd ~/tensorc-src
python3 - <<'EOF'
import sys, time
sys.path.insert(0, 'python')
import numpy as np
import tensorc as tc

# Tensores diminutos: todo corre en CPU (camino genérico), mide el coste
# host-side puro por op (Python, binding, creación de nodo de grafo).
x = tc.Tensor(np.random.rand(4, 4).astype(np.float32))
y = x
for _ in range(10):
    y = y.relu()
t0 = time.perf_counter()
N = 2000
for _ in range(N):
    y = y.relu()
print("host-side (tensor 4x4, CPU): %.3f ms/op" % ((time.perf_counter()-t0)/N*1000))

# Mismo tamaño que el LLM pero SIN GPU disponible para comparar cómputo:
# (no aplica aquí; solo interesa el suelo host-side)
# Y el coste de lanzamiento GPU puro: tensor residente, cadena larga.
big = tc.Tensor(np.random.rand(3072, 64).astype(np.float32))
z = big
for _ in range(10):
    z = z.relu()
t0 = time.perf_counter()
N = 1000
for _ in range(N):
    z = z.relu()
print("residente GPU (196K elems): %.3f ms/op" % ((time.perf_counter()-t0)/N*1000))
EOF
