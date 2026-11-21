"""TensorC: neural network framework written in C11, with autograd.

The numerical core is a C library; this package is a thin CPython
extension that wraps it. Tensors keep their parents alive automatically,
so ``backward()`` can be called from any scalar loss.
"""

from ._tensorc import (
    Adam,
    Conv2DLayer,
    DenseLayer,
    FlattenLayer,
    LSTMLayer,
    MaxPool2DLayer,
    ReLULayer,
    Sequential,
    Tensor,
    TransformerBlock,
    cuda_available,
    cuda_device,
    cuda_set_device,
    load,
    preprocess_mnist_images,
    save,
    sinusoidal_positions,
)
from .model import Model
from .rl import DQNAgent
from .sklearn import TensorCClassifier

__version__ = "0.15.0"

__all__ = [
    "Tensor",
    "Conv2DLayer",
    "DenseLayer",
    "ReLULayer",
    "MaxPool2DLayer",
    "FlattenLayer",
    "Sequential",
    "LSTMLayer",
    "TransformerBlock",
    "Adam",
    "Model",
    "TensorCClassifier",
    "DQNAgent",
    "load",
    "save",
    "preprocess_mnist_images",
    "sinusoidal_positions",
    "cuda_available",
    "cuda_device",
    "cuda_set_device",
    "__version__",
]
