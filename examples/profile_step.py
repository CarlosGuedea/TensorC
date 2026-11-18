#!/usr/bin/env python3
"""Perfil rápido del paso de entrenamiento de code_llm."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import tensorc as tc
from code_llm import CodeGPT, CKPT

def main():
    vocab, dim, ff, heads, layers, seq, batch = 8192, 512, 2048, 8, 6, 256, 8
    model = CodeGPT(vocab, dim, ff, heads, layers, seq)
    opt = tc.Adam(model.params, 3e-4)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, vocab, size=100000).astype(np.uint16)

    for step in range(4):
        starts = rng.integers(0, len(tokens) - seq - 1, size=batch)
        xb = np.stack([tokens[s:s + seq] for s in starts]).astype(np.float32)
        yb = np.stack([tokens[s + 1:s + seq + 1] for s in starts]
                      ).astype(np.float32).reshape(-1)
        t0 = time.perf_counter()
        probs = model.forward(tc.Tensor(xb))
        t1 = time.perf_counter()
        loss = probs.cross_entropy_sparse(tc.Tensor(yb))
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        opt.step()
        t4 = time.perf_counter()
        _ = float(loss.data[0])
        t5 = time.perf_counter()
        print("paso %d: fwd %.0f  ce %.0f  bwd %.0f  adam %.0f  sync %.0f ms"
              % (step, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                 (t4 - t3) * 1e3, (t5 - t4) * 1e3))
        del probs, loss

if __name__ == "__main__":
    main()
