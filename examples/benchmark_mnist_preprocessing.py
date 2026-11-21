"""Compara la preparación MNIST con NumPy y con el preprocesador C de TensorC."""

from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import tensorc


def preprocess_python(images):
    images = images.astype(np.float32) / 255.0
    return np.ascontiguousarray(images[:, None, :, :], dtype=np.float32)


def preprocess_tensorc(images):
    return tensorc.preprocess_mnist_images(images)


def measure(function, images):
    start = perf_counter()
    result = function(images)
    elapsed = perf_counter() - start
    del result
    return elapsed


with np.load(Path.home() / ".keras/datasets/mnist.npz") as dataset:
    X = dataset["x_train"]

reference = preprocess_python(X[:8])
native = preprocess_tensorc(X[:8])
assert np.allclose(
    np.asarray(native.data, dtype=np.float32).reshape(reference.shape),
    reference,
    rtol=1e-6,
    atol=1e-7,
)
del reference, native

# Calienta ambas rutas antes de medir y alterna el orden de cada repetición.
measure(preprocess_python, X)
measure(preprocess_tensorc, X)
python_times = []
tensorc_times = []
for run in range(5):
    order = ((preprocess_python, python_times),
             (preprocess_tensorc, tensorc_times))
    if run % 2:
        order = order[::-1]
    for function, times in order:
        times.append(measure(function, X))

python_median = median(python_times)
tensorc_median = median(tensorc_times)
print(f"Imágenes: {len(X):,} de {X.shape[1]}×{X.shape[2]}")
print(f"Python + NumPy: {python_median:.4f} s")
print(f"TensorC en C:   {tensorc_median:.4f} s")
print(f"Diferencia:     {python_median - tensorc_median:.4f} s")
print(f"Aceleración:    {python_median / tensorc_median:.2f}×")
