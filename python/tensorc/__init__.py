"""TensorC: neural network framework written in C11, with autograd.

The numerical core is a C library; this package is a thin CPython
extension that wraps it. Tensors keep their parents alive automatically,
so ``backward()`` can be called from any scalar loss.
"""

from ._tensorc import Adam, Conv2DLayer, DenseLayer, Tensor

__version__ = "0.11.0"

__all__ = ["Tensor", "Conv2DLayer", "DenseLayer", "Adam", "__version__"]
