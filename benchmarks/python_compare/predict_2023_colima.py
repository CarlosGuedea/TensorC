#!/usr/bin/env python3
"""Predicción del año 2023 (TMAX/TMIN diarias) — TensorC vs NumPy.

Estación climatológica 6040 (Colima, CONAGUA): registros diarios 1950 a
jun-2024. Tarea: entrenar con todo lo anterior a 2023 y predecir cada día
de 2023 (teacher forcing: la ventana de cada día usa los valores reales
de los 30 días anteriores). Misma arquitectura, mismos datos, mismo
optimizador en ambos lados.

Modelo: ventana de 30 días x [TMAX, TMIN, log1p(PRECIP), sinDOY, cosDOY]
(150 entradas) -> 64 ReLU -> 32 ReLU -> 2 lineales (TMAX, TMIN del día
siguiente, estandarizadas), MSE + Adam.

Métricas en 2023 desestandarizadas a °C: MSE, MAE y R² por variable,
contra el baseline de persistencia (predecir el valor de ayer).

Uso:
    python benchmarks/python_compare/predict_2023_colima.py
"""

import os
import sys
import time
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))

import tensorc as tc

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "6040_estacion_colima.txt")
WINDOW = 30
EPOCHS = 300
LR = 1e-3
SEED = 42


# ------------------------------------------------------------ datos

def load_series():
    """Serie diaria continua [TMAX, TMIN, PRECIP] con interpolación."""
    raw = {}
    with open(DATA, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5 or not parts[0][:4].isdigit():
                continue
            fecha = parts[0]

            def num(v):
                try:
                    return float(v)
                except ValueError:
                    return np.nan

            raw[fecha] = (num(parts[1].strip()), num(parts[3].strip()),
                          num(parts[4].strip()))
    start = date.fromisoformat(min(raw))
    end = date.fromisoformat(max(raw))
    days = (end - start).days + 1
    series = np.full((days, 3), np.nan, dtype=np.float32)
    dates = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        dates.append(d)
        if d in raw:
            series[i] = raw[d]
    # PRECIP faltante -> 0; TMAX/TMIN -> interpolación lineal.
    series[:, 0] = np.nan_to_num(series[:, 0], nan=0.0)
    for col in (1, 2):
        idx = np.arange(days)
        good = ~np.isnan(series[:, col])
        series[:, col] = np.interp(idx, idx[good], series[:, col][good])
    return dates, series


def build_dataset(dates, series):
    """Ventanas deslizantes: X [n,150] -> y [n,2] (TMAX,TMIN del día
    siguiente). Devuelve índices de test (días objetivo en 2023)."""
    days = len(dates)
    doy = np.array([int(d[5:7]) * 30 + int(d[8:10]) for d in dates],
                   dtype=np.float32)
    angle = 2.0 * np.pi * doy / 365.25
    feats = np.stack([
        series[:, 1], series[:, 2], np.log1p(series[:, 0]),
        np.sin(angle), np.cos(angle)
    ], axis=1)  # [days, 5]
    n = days - WINDOW
    sw = np.lib.stride_tricks.sliding_window_view(feats, (WINDOW, 5))
    X = sw.reshape(len(sw), WINDOW * 5)[:n]
    Y = series[WINDOW:, 1:3]  # TMAX, TMIN del día siguiente
    target_dates = dates[WINDOW:]
    test_idx = np.array([i for i, d in enumerate(target_dates)
                         if d.startswith("2023")])
    train_idx = np.array([i for i, d in enumerate(target_dates)
                          if d < "2023"])
    return X, Y, train_idx, test_idx


# ----------------------------------------------------------- métricas

def report(name, y_true, y_pred, baseline):
    out = []
    for k, var in enumerate(("TMAX", "TMIN")):
        mse = float(np.mean((y_true[:, k] - y_pred[:, k]) ** 2))
        mae = float(np.mean(np.abs(y_true[:, k] - y_pred[:, k])))
        ss_res = float(np.sum((y_true[:, k] - y_pred[:, k]) ** 2))
        ss_tot = float(np.sum((y_true[:, k] - y_true[:, k].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        b_mae = float(np.mean(np.abs(y_true[:, k] - baseline[:, k])))
        out.append("  %s: MSE %6.3f | MAE %5.3f °C | R² %.4f | persistencia MAE %5.3f °C"
                   % (var, mse, mae, r2, b_mae))
    print("%s\n%s" % (name, "\n".join(out)))
    return out


# ----------------------------------------------------------- TensorC

def train_tensorc(X, Ytr_std):
    l1 = tc.DenseLayer(WINDOW * 5, 64, seed=SEED)
    l2 = tc.DenseLayer(64, 32, seed=SEED + 1)
    l3 = tc.DenseLayer(32, 2, seed=SEED + 2)
    model = tc.Model(
        lambda inp: l3.forward(l2.forward(l1.forward(inp).relu()).relu()),
        [l1.weight, l1.bias, l2.weight, l2.bias, l3.weight, l3.bias],
        loss="mse", learning_rate=LR)
    start = time.perf_counter()
    model.fit(tc.Tensor(X), tc.Tensor(Ytr_std), epochs=EPOCHS, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000.0
    return model, elapsed


# ------------------------------------------------------------ NumPy

def train_numpy(X, Ytr_std):
    shapes = [(WINDOW * 5, 64), (64, 32), (32, 2)]
    rng = np.random.default_rng(SEED)
    w = [(rng.uniform(-1, 1, s).astype(np.float32) *
          np.sqrt(6.0 / (s[0] + s[1]))) for s in shapes]
    b = [np.zeros(s[1], dtype=np.float32) for s in shapes]
    params = [w[0], b[0], w[1], b[1], w[2], b[2]]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = len(X)
    start = time.perf_counter()
    for step in range(1, EPOCHS + 1):
        z1 = X @ w[0] + b[0]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ w[1] + b[1]
        h2 = np.maximum(z2, 0)
        pred = h2 @ w[2] + b[2]
        g = 2.0 * (pred - Ytr_std) / (n * 2)
        gw2 = h2.T @ g
        gb2 = g.sum(axis=0)
        g_h2 = g @ w[2].T
        g_z2 = g_h2 * (z2 > 0)
        gw1 = h1.T @ g_z2
        gb1 = g_z2.sum(axis=0)
        g_h1 = g_z2 @ w[1].T
        g_z1 = g_h1 * (z1 > 0)
        gw0 = X.T @ g_z1
        gb0 = g_z1.sum(axis=0)
        grads = [gw0, gb0, gw1, gb1, gw2, gb2]
        for i in range(6):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
            params[i] -= (LR * (m[i] / (1 - beta1 ** step)) /
                          (np.sqrt(v[i] / (1 - beta2 ** step)) + eps))
    elapsed = (time.perf_counter() - start) * 1000.0
    def predict(Xq):
        h = np.maximum(np.maximum(Xq @ w[0] + b[0], 0) @ w[1] + b[1], 0)
        return h @ w[2] + b[2]
    return predict, elapsed


def main():
    dates, series = load_series()
    X, Y, train_idx, test_idx = build_dataset(dates, series)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    # Estandarización con estadísticas del entrenamiento.
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    y_mean, y_std = Y_train.mean(axis=0), Y_train.std(axis=0) + 1e-8
    Xtr = ((X_train - x_mean) / x_std).astype(np.float32)
    Ytr_std = ((Y_train - y_mean) / y_std).astype(np.float32)
    Xte = ((X_test - x_mean) / x_std).astype(np.float32)

    print("Estación 6040 Colima: %d días | %d train / %d test (2023)"
          % (len(dates), len(train_idx), len(test_idx)))
    print("MLP %d->64->32->2, MSE + Adam (lr %g), %d épocas, batch completo"
          % (WINDOW * 5, LR, EPOCHS))
    print("=" * 70)

    model, tc_ms = train_tensorc(Xtr, Ytr_std)
    pred_tc = (np.asarray(model.predict(tc.Tensor(Xte)).data)
               .reshape(len(Xte), 2) * y_std + y_mean)
    print("TensorC : %8.1f ms de entrenamiento" % tc_ms)
    report("TensorC", Y_test, pred_tc, np.roll(Y_test, 1, axis=0))

    predict_np, np_ms = train_numpy(Xtr, Ytr_std)
    pred_np = predict_np(Xte) * y_std + y_mean
    print("NumPy   : %8.1f ms de entrenamiento" % np_ms)
    report("NumPy", Y_test, pred_np, np.roll(Y_test, 1, axis=0))

    print("=" * 70)
    mse_tc = float(np.mean((Y_test - pred_tc) ** 2))
    mse_np = float(np.mean((Y_test - pred_np) ** 2))
    faster = "TensorC" if tc_ms < np_ms else "NumPy"
    better = "TensorC" if mse_tc < mse_np else "NumPy"
    print("Más rápido: %s (%.2fx)   Mejor MSE 2023: %s (%.4f vs %.4f)"
          % (faster, max(tc_ms, np_ms) / min(tc_ms, np_ms), better,
             min(mse_tc, mse_np), max(mse_tc, mse_np)))


if __name__ == "__main__":
    main()
