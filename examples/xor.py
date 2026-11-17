"""XOR benchmark matching examples/xor.c; run: python3 examples/xor.py [pure|numpy]."""
import sys
import time

EPOCHS = 12_000
LR = 1.0
X = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
Y = [0.0, 1.0, 1.0, 0.0]


def sigmoid(x):
    if x >= 0:
        z = __import__("math").exp(-x)
        return 1.0 / (1.0 + z)
    z = __import__("math").exp(x)
    return z / (1.0 + z)


def run_pure():
    # 2 -> 8 -> 1; same online SGD/MSE/backprop sequence as the C example.
    import random
    rng = random.Random(42)
    limit1 = (6.0 / 10.0) ** 0.5
    limit2 = (6.0 / 9.0) ** 0.5
    w1 = [[rng.uniform(-limit1, limit1) for _ in range(8)] for _ in range(2)]
    b1 = [0.0] * 8
    w2 = [[rng.uniform(-limit2, limit2)] for _ in range(8)]
    b2 = [0.0]
    loss = 0.0
    start = time.perf_counter()
    for _ in range(EPOCHS):
        total = 0.0
        for (x0, x1), target in zip(X, Y):
            a0 = [x0, x1]
            z1 = [b1[j] + a0[0] * w1[0][j] + a0[1] * w1[1][j] for j in range(8)]
            a1 = [sigmoid(v) for v in z1]
            z2 = b2[0] + sum(a1[j] * w2[j][0] for j in range(8))
            a2 = sigmoid(z2)
            error = a2 - target
            total += error * error
            d2 = error * a2 * (1.0 - a2)
            d1 = [w2[j][0] * d2 * a1[j] * (1.0 - a1[j]) for j in range(8)]
            for j in range(8):
                w2[j][0] -= LR * a1[j] * d2
                b1[j] -= LR * d1[j]
            for i in range(2):
                for j in range(8):
                    w1[i][j] -= LR * a0[i] * d1[j]
            b2[0] -= LR * d2
        loss = total / len(X)
    elapsed = time.perf_counter() - start
    predictions = []
    for (x0, x1) in X:
        hidden = [sigmoid(x0 * w1[0][j] + x1 * w1[1][j] + b1[j]) for j in range(8)]
        predictions.append(sigmoid(sum(hidden[j] * w2[j][0] for j in range(8)) + b2[0]))
    return elapsed, loss, predictions


def run_numpy():
    import numpy as np
    rng = np.random.default_rng(42)
    w1 = rng.uniform(-(6 / 10) ** 0.5, (6 / 10) ** 0.5, (2, 8))
    b1 = np.zeros(8)
    w2 = rng.uniform(-(6 / 9) ** 0.5, (6 / 9) ** 0.5, (8, 1))
    b2 = np.zeros(1)
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64).reshape(4, 1)
    loss = 0.0
    start = time.perf_counter()
    for _ in range(EPOCHS):
        total = 0.0
        for row in range(4):
            a0 = x[row]
            z1 = a0 @ w1 + b1
            a1 = 1.0 / (1.0 + np.exp(-z1))
            z2 = a1 @ w2 + b2
            a2 = 1.0 / (1.0 + np.exp(-z2))
            error = a2 - y[row]
            total += float(np.mean(error * error))
            d2 = error * a2 * (1.0 - a2)
            d1 = (w2 @ d2) * a1 * (1.0 - a1)
            w2 -= LR * np.outer(a1, d2)
            b2 -= LR * d2
            w1 -= LR * np.outer(a0, d1)
            b1 -= LR * d1
        loss = total / 4.0
    elapsed = time.perf_counter() - start
    hidden = 1.0 / (1.0 + np.exp(-(x @ w1 + b1)))
    predictions = (1.0 / (1.0 + np.exp(-(hidden @ w2 + b2)))).ravel().tolist()
    return elapsed, loss, predictions


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pure"
    if mode == "pure":
        elapsed, loss, predictions = run_pure()
        label = "Python puro"
    elif mode == "numpy":
        elapsed, loss, predictions = run_numpy()
        label = "Python + NumPy"
    else:
        raise SystemExit("Uso: python3 examples/xor.py [pure|numpy]")
    print(f"{label}: {EPOCHS} épocas, {EPOCHS * len(X)} actualizaciones SGD")
    print(f"Tiempo de entrenamiento: {elapsed:.6f} s | MSE final: {loss:.6f}")
    print("Salidas XOR:", " ".join(f"{p:.3f}" for p in predictions))


if __name__ == "__main__":
    main()
