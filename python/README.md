# tensorc (Python)

Bindings de Python para [TensorC](../README.md): el núcleo numérico es la
biblioteca en C11 (autograd, tensores N-dimensionales, Conv2D con
stride/padding, max pooling, softmax/cross-entropy, SGD y Adam); este
paquete solo expone esa funcionalidad a Python mediante una extensión
CPython escrita en C.

## Uso

```python
import tensorc

x = tensorc.Tensor([1, 2, 3, 4], shape=(2, 2), requires_grad=True)
w = tensorc.Tensor([1.0, 0.0, 0.0, 1.0], shape=(2, 2), requires_grad=True)
loss = x.matmul(w).relu().sum()
loss.backward()
x.sgd_step(0.01)

capa = tensorc.Conv2DLayer(1, 8, 3, 3, stride=(1, 1), padding=(1, 1), seed=42)
features = capa.forward(imagen)  # conv + sesgo, con autograd
```

## Compilar e instalar en local

Requisitos: Python 3.9+ con cabeceras de desarrollo (`python3-dev` en
Linux, Python de Homebrew en macOS ya las incluye).

```sh
python3 -m pip install ./python
```

## Publicar en PyPI

1. Genera una distribución de código fuente y, si puedes, ruedas
   precompiladas con [cibuildwheel](https://cibuildwheel.readthedocs.io/)
   para macOS/Linux/Windows:

   ```sh
   python3 -m pip install build cibuildwheel twine
   cd python
   python3 -m build          # sdist + wheel de la plataforma actual
   ```

2. Sube el paquete (necesitas cuenta en PyPI y un token):

   ```sh
   python3 -m twine upload dist/*
   ```

Después cualquiera puede instalarlo con `pip install tensorc`.

## Diseño

- `tensorc/_tensorc.c` — extensión CPython (C11). Convierte entre objetos
  Python y `tc_tensor_nd`; no contiene matemáticas.
- `minitf.c` / `ndtensor.c` — el núcleo en C, idéntico al que usa CMake.
- En macOS se enlaza Apple Accelerate con la misma ruta im2col+GEMM que el
  build de CMake; en Linux/Windows se usa el backend genérico.
