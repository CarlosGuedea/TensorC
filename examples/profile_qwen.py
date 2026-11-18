"""Perfil rápido: tiempo de forward y backward por separado + pico de VRAM."""
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from qwen_tensorc import load_model

import tensorc as tc

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = int(sys.argv[2]) if len(sys.argv) > 2 else 256

peak = [0]
stop = False

def watch():
    smi = "/usr/lib/wsl/lib/nvidia-smi"
    while not stop:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"]).decode()
            mem, util = [int(x.strip()) for x in out.split(",")]
            peak[0] = max(peak[0], mem)
        except Exception:
            pass
        time.sleep(0.5)

threading.Thread(target=watch, daemon=True).start()

trainable = {"ln_f"}
for i in range(20, 24):
    trainable |= {f"layers.{i}.attn", f"layers.{i}.mlp",
                  f"layers.{i}.ln1", f"layers.{i}.ln2"}

t0 = time.perf_counter()
model = load_model(max_seq=T, trainable=trainable)
print(f"carga: {time.perf_counter() - t0:.1f}s  vram_pico={peak[0]} MiB",
      flush=True)

rng = np.random.default_rng(0)
xb = rng.integers(0, 150000, size=(B, T)).astype(np.float32)
yb = rng.integers(0, 150000, size=(B * T,)).astype(np.float32)

t0 = time.perf_counter()
loss = model.loss(tc.Tensor(xb), tc.Tensor(yb))
print(f"forward+loss: {time.perf_counter() - t0:.1f}s  loss={float(loss.data[0]):.3f}  vram_pico={peak[0]} MiB", flush=True)

t0 = time.perf_counter()
loss.backward()
print(f"backward: {time.perf_counter() - t0:.1f}s  vram_pico={peak[0]} MiB",
      flush=True)

opt = tc.Adam(model.params, 1e-5)
t0 = time.perf_counter()
opt.step()
print(f"adam: {time.perf_counter() - t0:.2f}s  vram_pico={peak[0]} MiB",
      flush=True)
stop = True
