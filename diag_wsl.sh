#!/bin/bash
# Micro-benchmark: coste por op con datos ya residentes en device.
set -e
export PATH="$HOME/tensorc-env/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/tensorc-env/lib:$LD_LIBRARY_PATH"
cd ~/tensorc-src
python3 - <<'EOF'
import sys, time
sys.path.insert(0, 'python')
import numpy as np
import tensorc as tc

x = tc.Tensor(np.random.rand(3072, 64).astype(np.float32))
# warmup + residencia
y = x
for _ in range(20):
    y = y.relu()
_ = np.asarray(y.data)

for name, fn in [
    ("relu   196K elems", lambda v: v.relu()),
    ("add    196K elems", lambda v: v.add(v)),
    ("matmul 3072x64x64", lambda v: v.matmul(w)),
]:
    w = tc.Tensor(np.random.rand(64, 64).astype(np.float32))
    y = x
    for _ in range(10):
        y = fn(y)
    t0 = time.perf_counter()
    N = 300
    for _ in range(N):
        y = fn(y)
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"{name}: {dt:.3f} ms/op")
EOF
