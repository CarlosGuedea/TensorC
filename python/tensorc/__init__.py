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
    load,
    save,
    sinusoidal_positions,
)
from .model import Model

__version__ = "0.14.0"

__all__ = [
    "Tensor",
    "Conv2DLayer",
    "DenseLayer",
    "LSTMLayer",
    "TransformerBlock",
    "Adam",
    "Model",
    "load",
    "save",
    "sinusoidal_positions",
    "__version__",
]
