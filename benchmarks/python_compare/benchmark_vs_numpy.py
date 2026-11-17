#!/usr/bin/env python3
"""Benchmark reproducible: TensorC vs NumPy en redes neuronales.

Mismos datos, mismas dimensiones, mismas épocas. Los tiempos se reportan
en milisegundos. Ambas rutas usan float32 y aprovechan BLAS del sistema
(Accelerate/OpenBLAS), así que la comparación es framework contra NumPy,
no BLAS contra BLAS puro.

Uso:
    python benchmarks/python_compare/benchmark_vs_numpy.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))

import tensorc as tc


def ms(seconds):
    return seconds * 1000.0


def timeit(fn, repeats=5):
    """Devuelve (mediana en ms) de `repeats` ejecuciones completas."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(ms(time.perf_counter() - start))
    samples.sort()
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------
# 1) Entrenamiento de un MLP: 784 -> 128 (ReLU) -> 10, MSE, Adam.
# ---------------------------------------------------------------------

N, IN, HID, OUT, EPOCHS = 1024, 784, 128, 10, 30


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, IN), dtype=np.float32)
    w_true = rng.standard_normal((IN, OUT), dtype=np.float32)
    y = (x @ w_true).astype(np.float32)
    return x, y


def mlp_tensorc(x, y):
    l1 = tc.DenseLayer(IN, HID, seed=1)
    l2 = tc.DenseLayer(HID, OUT, seed=2)
    model = tc.Model(lambda inp: l2.forward(l1.forward(inp).relu()),
                     [l1.weight, l1.bias, l2.weight, l2.bias],
                     loss="mse", learning_rate=1e-3)
    model.fit(tc.Tensor(x), tc.Tensor(y), epochs=EPOCHS, verbose=False)
    return model.evaluate(tc.Tensor(x), tc.Tensor(y))


def mlp_numpy(x, y):
    rng = np.random.default_rng(7)
    w1 = rng.standard_normal((IN, HID), dtype=np.float32) * 0.05
    b1 = np.zeros(HID, dtype=np.float32)
    w2 = rng.standard_normal((HID, OUT), dtype=np.float32) * 0.05
    b2 = np.zeros(OUT, dtype=np.float32)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8
    for step in range(1, EPOCHS + 1):
        # forward
        z1 = x @ w1 + b1
        h1 = np.maximum(z1, 0)
        pred = h1 @ w2 + b2
        # MSE backward (d/dpred de mean((pred-y)^2))
        g_pred = 2.0 * (pred - y) / pred.size
        gw2 = h1.T @ g_pred
        gb2 = g_pred.sum(axis=0)
        g_h1 = g_pred @ w2.T
        g_z1 = g_h1 * (z1 > 0)
        gw1 = x.T @ g_z1
        gb1 = g_z1.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2]
        # Adam
        for i in range(4):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
            m_hat = m[i] / (1 - beta1 ** step)
            v_hat = v[i] / (1 - beta2 ** step)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    pred = np.maximum(x @ w1 + b1, 0) @ w2 + b2
    return float(np.mean((pred - y) ** 2))


# ---------------------------------------------------------------------
# 2) Inferencia del MLP entrenado: 200 pases forward completos.
# ---------------------------------------------------------------------
INFER_REPEATS = 200


def infer_tensorc(x):
    l1 = tc.DenseLayer(IN, HID, seed=1)
    l2 = tc.DenseLayer(HID, OUT, seed=2)

    def run():
        for _ in range(INFER_REPEATS):
            out = l2.forward(l1.forward(x).relu())
    return run


def infer_numpy(x, w1, b1, w2, b2):
    def run():
        for _ in range(INFER_REPEATS):
            out = np.maximum(x @ w1 + b1, 0) @ w2 + b2
    return run


# ---------------------------------------------------------------------
# 3) Conv2D forward: [8,3,32,32] * [16,3,3,3].
# ---------------------------------------------------------------------
CN, CC, CH, CW = 8, 3, 32, 32
KN, KH, KW = 16, 3, 3
CONV_REPEATS = 20


def conv_tensorc(x, k):
    xt = tc.Tensor(x)
    kt = tc.Tensor(k)

    def run():
        for _ in range(CONV_REPEATS):
            out = xt.conv2d(kt)
    return run


def conv_numpy(x, k):
    oh, ow = CH - KH + 1, CW - KW + 1

    def run():
        for _ in range(CONV_REPEATS):
            col = np.lib.stride_tricks.sliding_window_view(
                x, (KH, KW), axis=(2, 3))            # [N,C,OH,OW,KH,KW]
            col = col.transpose(0, 2, 3, 1, 4, 5)    # [N,OH,OW,C,KH,KW]
            col = col.reshape(CN * oh * ow, CC * KH * KW)
            out = col @ k.reshape(KN, CC * KH * KW).T
            out = out.reshape(CN, oh, ow, KN).transpose(0, 3, 1, 2)
    return run


def main():
    x, y = make_data()
    print("=" * 64)
    print("TensorC %s vs NumPy %s  (mismo datos, mismas dimensiones)"
          % (tc.__version__, np.__version__))
    print("=" * 64)

    tc_ms = timeit(lambda: mlp_tensorc(x, y))
    np_ms = timeit(lambda: mlp_numpy(x, y))
    print("\n[1] MLP 784->128->10, MSE + Adam, %d épocas, batch %d"
          % (EPOCHS, N))
    print("    TensorC: %8.1f ms      NumPy: %8.1f ms   (%.1fx)"
          % (tc_ms, np_ms, np_ms / tc_ms))

    w1 = np.random.default_rng(7).standard_normal((IN, HID)).astype(np.float32)
    xt = tc.Tensor(x)
    ti_ms = timeit(infer_tensorc(xt))
    ni_ms = timeit(infer_numpy(x, w1, np.zeros(HID, np.float32),
                               np.random.default_rng(8)
                               .standard_normal((HID, OUT)).astype(np.float32),
                               np.zeros(OUT, np.float32)))
    print("\n[2] Inferencia MLP: %d forwards de [%d,%d]" % (INFER_REPEATS, N, IN))
    print("    TensorC: %8.1f ms      NumPy: %8.1f ms   (%.1fx)"
          % (ti_ms, ni_ms, ni_ms / ti_ms))

    rng = np.random.default_rng(3)
    cx = rng.standard_normal((CN, CC, CH, CW), dtype=np.float32)
    ck = rng.standard_normal((KN, CC, KH, KW), dtype=np.float32)
    tconv_ms = timeit(conv_tensorc(cx, ck), repeats=3)
    nconv_ms = timeit(conv_numpy(cx, ck), repeats=3)
    print("\n[3] Conv2D forward: [%d,%d,%d,%d] * [%d,%d,%d,%d], %d pasadas"
          % (CN, CC, CH, CW, KN, CC, KH, KW, CONV_REPEATS))
    print("    TensorC: %8.1f ms      NumPy: %8.1f ms   (%.1fx)"
          % (tconv_ms, nconv_ms, nconv_ms / tconv_ms))
    print()


if __name__ == "__main__":
    main()
