#!/usr/bin/env python3
"""TensorC vs NumPy en el dataset real Pima Indians Diabetes (768×8, binario).

Misma arquitectura (8 -> 16 ReLU -> 2 softmax), misma loss (cross-entropy),
mismo optimizador (Adam, lr 0.01, betas 0.9/0.999), mismas épocas, mismos
datos: estandarización y split 80/20 con semilla fija. Reporta tiempos de
entrenamiento en ms y exactitud (accuracy) sobre el conjunto de prueba.

Uso:
    python benchmarks/python_compare/pima_diabetes.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))

import tensorc as tc

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "pima_diabetes.csv")
EPOCHS = 300
LR = 0.01
SEED = 42


def load_data():
    with open(DATA, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], np.array([[float(v) for v in r] for r in rows[1:]],
                                     dtype=np.float32)
    x, y = data[:, :8], data[:, 8].astype(np.int64)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    # Estandarización con estadísticas del entrenamiento.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    return ((x_train - mean) / std, y_train,
            (x_test - mean) / std, y_test)


def one_hot(y, classes=2):
    out = np.zeros((len(y), classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def accuracy(probabilities, labels):
    predicted = np.argmax(np.asarray(probabilities), axis=1)
    return float(np.mean(predicted == labels))


# ------------------------------------------------------------------ TensorC

def train_tensorc(x_train, y_train_oh):
    l1 = tc.DenseLayer(8, 16, seed=SEED)
    l2 = tc.DenseLayer(16, 2, seed=SEED + 1)
    model = tc.Model(lambda inp: l2.forward(l1.forward(inp).relu()).softmax(),
                     [l1.weight, l1.bias, l2.weight, l2.bias],
                     loss="cross_entropy", learning_rate=LR)
    start = time.perf_counter()
    history = model.fit(tc.Tensor(x_train), tc.Tensor(y_train_oh),
                        epochs=EPOCHS, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000.0
    return model, history["loss"][-1], elapsed


def predict_tensorc(model, x):
    return np.asarray(model.predict(tc.Tensor(x)).data).reshape(len(x), 2)


# ------------------------------------------------------------------- NumPy

def train_numpy(x_train, y_train_oh):
    rng = np.random.default_rng(SEED)
    w1 = (rng.uniform(-1, 1, (8, 16)).astype(np.float32) *
          np.sqrt(6.0 / (8 + 16)))
    b1 = np.zeros(16, dtype=np.float32)
    w2 = (rng.uniform(-1, 1, (16, 2)).astype(np.float32) *
          np.sqrt(6.0 / (16 + 2)))
    b2 = np.zeros(2, dtype=np.float32)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = len(x_train)
    start = time.perf_counter()
    for step in range(1, EPOCHS + 1):
        # forward: softmax(z2), cross-entropy contra one-hot
        z1 = x_train @ w1 + b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ w2 + b2
        z2 -= z2.max(axis=1, keepdims=True)
        exp = np.exp(z2)
        probs = exp / exp.sum(axis=1, keepdims=True)
        # gradiente de CE+softmax: (probs - one_hot) / n
        g_z2 = (probs - y_train_oh) / n
        gw2 = h1.T @ g_z2
        gb2 = g_z2.sum(axis=0)
        g_h1 = g_z2 @ w2.T
        g_z1 = g_h1 * (z1 > 0)
        gw1 = x_train.T @ g_z1
        gb1 = g_z1.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2]
        for i in range(4):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
            params[i] -= (LR * (m[i] / (1 - beta1 ** step)) /
                          (np.sqrt(v[i] / (1 - beta2 ** step)) + eps))
        # pérdida final de la época
        z1 = x_train @ w1 + b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ w2 + b2
        z2 -= z2.max(axis=1, keepdims=True)
        exp = np.exp(z2)
        probs = exp / exp.sum(axis=1, keepdims=True)
        p = np.clip(probs[np.arange(n), y_train_oh.argmax(axis=1)], 1e-7, 1)
        loss = float(-np.log(p).mean())
    elapsed = (time.perf_counter() - start) * 1000.0
    state = params
    return state, loss, elapsed


def predict_numpy(state, x):
    w1, b1, w2, b2 = state
    z1 = np.maximum(x @ w1 + b1, 0)
    z2 = z1 @ w2 + b2
    z2 -= z2.max(axis=1, keepdims=True)
    exp = np.exp(z2)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    x_train, y_train, x_test, y_test = load_data()
    y_train_oh = one_hot(y_train)
    print("Pima Indians Diabetes: %d train / %d test, 8 features, 2 classes"
          % (len(x_train), len(x_test)))
    print("MLP 8->16(ReLU)->2(softmax), cross-entropy + Adam, %d epochs"
          % EPOCHS)
    print("=" * 64)

    tc_model, tc_loss, tc_ms = train_tensorc(x_train, y_train_oh)
    tc_acc = accuracy(predict_tensorc(tc_model, x_test), y_test)
    print("TensorC : %7.1f ms   loss final %.4f   accuracy test %.4f"
          % (tc_ms, tc_loss, tc_acc))

    np_state, np_loss, np_ms = train_numpy(x_train, y_train_oh)
    np_acc = accuracy(predict_numpy(np_state, x_test), y_test)
    print("NumPy   : %7.1f ms   loss final %.4f   accuracy test %.4f"
          % (np_ms, np_loss, np_acc))

    print("=" * 64)
    faster = "TensorC" if tc_ms < np_ms else "NumPy"
    better = "TensorC" if tc_acc > np_acc else ("NumPy" if np_acc > tc_acc else "empate")
    print("Más rápido: %s (%.2fx)   Más certero: %s (Δ %.4f)"
          % (faster, max(tc_ms, np_ms) / min(tc_ms, np_ms), better,
             abs(tc_acc - np_acc)))


if __name__ == "__main__":
    main()
