"""TensorC: neural network framework written in C11, with autograd.

The numerical core is a C library; this package is a thin CPython
extension that wraps it. Tensors keep their parents alive automatically,
so ``backward()`` can be called from any scalar loss.
"""

from ._tensorc import (
    Adam,
    Conv2DLayer,
    DenseLayer,
    LSTMLayer,
    Tensor,
    TransformerBlock,
    cuda_available,
    cuda_device,
    cuda_set_device,
    load,
    save,
    sinusoidal_positions,
)
from .model import Model
from .sklearn import TensorCClassifier

__version__ = "0.15.0"

__all__ = [
    "Tensor",
    "Conv2DLayer",
    "DenseLayer",
    "LSTMLayer",
    "TransformerBlock",
    "Adam",
    "Model",
    "TensorCClassifier",
    "load",
    "save",
    "sinusoidal_positions",
    "cuda_available",
    "cuda_device",
    "cuda_set_device",
    "__version__",
]
