"""Compara una CNN MNIST NumPy con el entrenamiento nativo C de TensorC."""

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorc


TRAIN_SAMPLES = 8192
TEST_SAMPLES = 2048
BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = np.float32(0.001)


def conv_forward(x, weight, bias):
    kh, kw = weight.shape[2:]
    x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (kh, kw), axis=(2, 3))
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, weight.shape[1] * kh * kw)
    out = (cols @ weight.reshape(weight.shape[0], -1).T).reshape(
        x.shape[0], x.shape[2], x.shape[3], weight.shape[0]
    ).transpose(0, 3, 1, 2)
    out += bias
    return out, (x_pad, cols, weight.shape)


def conv_backward(grad, cache, weight):
    x_pad, cols, weight_shape = cache
    n, channels, height, width = x_pad.shape
    filters, _, kh, kw = weight_shape
    grad_rows = grad.transpose(0, 2, 3, 1).reshape(-1, filters)
    dw = (grad_rows.T @ cols).reshape(weight_shape)
    db = grad.sum(axis=(0, 2, 3), keepdims=True)
    dcols = (grad_rows @ weight.reshape(filters, -1)).reshape(
        n, height - kh + 1, width - kw + 1, channels, kh, kw
    ).transpose(0, 3, 4, 5, 1, 2)
    dx_pad = np.zeros_like(x_pad)
    out_h, out_w = dcols.shape[-2:]
    for row in range(kh):
        for col in range(kw):
            dx_pad[:, :, row:row + out_h, col:col + out_w] += dcols[:, :, row, col]
    return dx_pad[:, :, 1:-1, 1:-1], dw, db


def pool_forward(x):
    n, channels, height, width = x.shape
    windows = x.reshape(n, channels, height // 2, 2, width // 2, 2)
    windows = windows.transpose(0, 1, 2, 4, 3, 5).reshape(
        n, channels, height // 2, width // 2, 4
    )
    indices = windows.argmax(axis=-1)
    return windows.max(axis=-1), indices


def pool_backward(grad, indices):
    dx = np.zeros((grad.shape[0], grad.shape[1], grad.shape[2] * 2,
                   grad.shape[3] * 2), dtype=np.float32)
    for index in range(4):
        dx[:, :, index // 2::2, index % 2::2] += grad * (indices == index)
    return dx


def forward(x, params, cache=False):
    w1, b1, w2, b2, wd, bd = params
    c1, conv1 = conv_forward(x, w1, b1)
    r1 = np.maximum(c1, 0)
    p1, pool1 = pool_forward(r1)
    c2, conv2 = conv_forward(p1, w2, b2)
    r2 = np.maximum(c2, 0)
    p2, pool2 = pool_forward(r2)
    flat = p2.reshape(len(x), -1)
    logits = flat @ wd + bd
    if cache:
        return logits, (conv1, c1, pool1, conv2, c2, pool2, flat)
    return logits


def train_numpy(x, labels, params):
    moments = [np.zeros_like(p) for p in params]
    variances = [np.zeros_like(p) for p in params]
    state = 42
    step = 0
    losses = []
    for _ in range(EPOCHS):
        order = np.arange(len(x))
        for i in range(len(order), 1, -1):
            state ^= (state << 13) & 0xFFFFFFFF
            state &= 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            state &= 0xFFFFFFFF
            j = state % i
            order[i - 1], order[j] = order[j], order[i - 1]
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            xb, yb = x[indices], labels[indices]
            logits, saved = forward(xb, params, cache=True)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs /= probs.sum(axis=1, keepdims=True)
            total -= np.log(np.maximum(probs[np.arange(len(yb)), yb], 1e-30)).sum()
            grad = probs
            grad[np.arange(len(yb)), yb] -= 1
            grad /= len(yb)
            conv1, c1, pool1, conv2, c2, pool2, flat = saved
            grads = [None] * len(params)
            grads[4] = flat.T @ grad
            grads[5] = grad.sum(axis=0, keepdims=True)
            grad = (grad @ params[4].T).reshape(
                len(yb), 16, 7, 7
            )
            grad = pool_backward(grad, pool2)
            grad[c2 <= 0] = 0
            grad, grads[2], grads[3] = conv_backward(grad, conv2, params[2])
            grad = pool_backward(grad, pool1)
            grad[c1 <= 0] = 0
            _, grads[0], grads[1] = conv_backward(grad, conv1, params[0])

            step += 1
            bias1 = np.float32(1) - np.float32(0.9) ** np.float32(step)
            bias2 = np.float32(1) - np.float32(0.999) ** np.float32(step)
            for index, (parameter, derivative) in enumerate(zip(params, grads)):
                moments[index] *= np.float32(0.9)
                moments[index] += np.float32(0.1) * derivative
                variances[index] *= np.float32(0.999)
                variances[index] += np.float32(0.001) * derivative * derivative
                m_hat = moments[index] / bias1
                v_hat = variances[index] / bias2
                parameter -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + np.float32(1e-8))
            seen += len(yb)
        losses.append(total / seen)
    return losses


def predict_numpy(x, params):
    results = []
    for start in range(0, len(x), BATCH_SIZE):
        logits = forward(x[start:start + BATCH_SIZE], params)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        results.append(probabilities / probabilities.sum(axis=1, keepdims=True))
    return np.concatenate(results)


with np.load(Path.home() / ".keras/datasets/mnist.npz") as dataset:
    rng = np.random.default_rng(2026)
    train_ids = rng.choice(len(dataset["x_train"]), TRAIN_SAMPLES, replace=False)
    test_ids = rng.choice(len(dataset["x_test"]), TEST_SAMPLES, replace=False)
    raw_train, y_train = dataset["x_train"][train_ids], dataset["y_train"][train_ids]
    raw_test, y_test = dataset["x_test"][test_ids], dataset["y_test"][test_ids]

x_train_numpy = np.ascontiguousarray(raw_train[:, None].astype(np.float32) / 255.0)
x_test_numpy = np.ascontiguousarray(raw_test[:, None].astype(np.float32) / 255.0)
x_train_c = tensorc.preprocess_mnist_images(raw_train)
x_test_c = tensorc.preprocess_mnist_images(raw_test)

network = tensorc.Sequential([
    tensorc.Conv2DLayer(1, 8, 3, 3, padding=(1, 1), seed=101),
    tensorc.ReLULayer(),
    tensorc.MaxPool2DLayer(2, 2),
    tensorc.Conv2DLayer(8, 16, 3, 3, padding=(1, 1), seed=202),
    tensorc.ReLULayer(),
    tensorc.MaxPool2DLayer(2, 2),
    tensorc.FlattenLayer(),
    tensorc.DenseLayer(16 * 7 * 7, 10, seed=303),
])
numpy_params = [
    np.asarray(param.data, dtype=np.float32).reshape(param.shape).copy()
    for param in network.params
]
sample_c = tensorc.preprocess_mnist_images(raw_train[:8])
assert np.allclose(
    np.asarray(sample_c.data, dtype=np.float32).reshape(8, 1, 28, 28),
    x_train_numpy[:8], rtol=1e-6, atol=1e-7,
)
assert np.allclose(
    np.asarray(network.forward(tensorc.Tensor(x_train_numpy[:8])).data).reshape(8, 10),
    forward(x_train_numpy[:8], numpy_params), rtol=3e-4, atol=3e-5,
)

classifier = tensorc.TensorCClassifier(
    model=network, epochs=EPOCHS, batch_size=BATCH_SIZE,
    learning_rate=float(LEARNING_RATE), random_state=42, verbose=False,
)
start = perf_counter()
classifier.fit(x_train_c, y_train)
c_training = perf_counter() - start
start = perf_counter()
c_probabilities = classifier.predict_proba(x_test_c)
c_inference = perf_counter() - start
c_accuracy = float((np.argmax(c_probabilities, axis=1) == y_test).mean())

start = perf_counter()
numpy_losses = train_numpy(x_train_numpy, y_train, numpy_params)
numpy_training = perf_counter() - start
start = perf_counter()
numpy_probabilities = predict_numpy(x_test_numpy, numpy_params)
numpy_inference = perf_counter() - start
numpy_accuracy = float((np.argmax(numpy_probabilities, axis=1) == y_test).mean())

print(f"MNIST: {TRAIN_SAMPLES} train / {TEST_SAMPLES} prueba, {EPOCHS} épocas, lote {BATCH_SIZE}")
print(f"TensorC C: entrenamiento {c_training:.3f} s | inferencia {c_inference:.3f} s | accuracy {c_accuracy:.2%}")
print(f"Python/NumPy: entrenamiento {numpy_training:.3f} s | inferencia {numpy_inference:.3f} s | accuracy {numpy_accuracy:.2%}")
print(f"Aceleración entrenamiento TensorC: {numpy_training / c_training:.2f}×")
print(f"Imágenes/seg inferencia: C {TEST_SAMPLES / c_inference:.0f} | NumPy {TEST_SAMPLES / numpy_inference:.0f}")
print(f"Diferencia de accuracy: {abs(c_accuracy - numpy_accuracy):.2%}")
