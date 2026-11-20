# tensorc

Consulta la [guía detallada de uso](../GUIA_USO.md) para instalación,
entrenamiento, CNN, MNIST, C11, el wrapper estilo scikit-learn y solución de
errores frecuentes.

Framework de redes neuronales escrito en **C11** con autodiferenciación,
invocable desde Python. El núcleo numérico es una biblioteca C pura
(sin dependencias externas; en macOS usa Apple Accelerate) y este paquete
es una extensión CPython escrita en C que la expone: **sin C++, sin
ctypes, sin overhead de FFI genérico**.

```python
import tensorc

x = tensorc.Tensor([1, 2, 3, 4], shape=(2, 2), requires_grad=True)
w = tensorc.Tensor([1, 0, 0, 1], shape=(2, 2), requires_grad=True)
loss = x.matmul(w).relu().sum()
loss.backward()
x.sgd_step(0.01)
```

## Instalación

```sh
pip install tensorc
```

Ruedas precompiladas (no requieren compilador) para **CPython 3.9–3.13**
en macOS (Intel y Apple Silicon), Linux (x86_64 y aarch64) y Windows
(AMD64). En cualquier otra plataforma pip compila desde el código fuente
automáticamente; solo necesitas un compilador C11.

## Qué incluye

- **Tensores N-dimensionales con autograd**: suma, multiplicación
  elemento a elemento con broadcasting, matmul 2D, ReLU, sigmoid, tanh,
  softmax, suma, reshape, slice en eje, concat N-aria, transposición 2D,
  permutación general de ejes, layer norm (sobre la última dimensión),
  max pooling 2D y average pooling 2D, todas con gradientes analíticos.
- **Pérdidas y optimizadores**: MSE, cross-entropy (gradiente
  `(p − t) / filas`), SGD por paso y Adam sobre listas de parámetros.
- **Convolución 2D** con stride y padding configurables; formatos
  `[N, C, H, W]` × kernel `[K, C, KH, KW]`; ruta im2col+GEMM (Accelerate)
  para convoluciones grandes.
- **Capas de alto nivel entrenables**: `DenseLayer`, `Conv2DLayer`,
  `LSTMLayer` (secuencias time-major `[seq, batch, in]`) y
  `TransformerBlock` (atención multi-cabeza + conexiones residuales +
  FFN, pre-norm).
- **Atención**: single-head (`Tensor.attention`, con máscara causal
  opcional) y multi-cabeza (`Tensor.multihead_attention`), más
  codificación posicional sinusoidal (`sinusoidal_positions`).
- **Entrenamiento de alto nivel**: `tensorc.Model` con `fit` (mini-batches
  con shuffle), `evaluate`, `predict`, `save_weights` y `load_weights`.
- **Serialización**: guardar/cargar tensores y checkpoints binarios de
  parámetros (`tensorc.save`, `tensorc.load`, `Tensor.save`,
  `Tensor.copy_`), con validación de formato y formas.

Cada operación valida formas incompatibles y reporta el error como
excepción de Python.

## Ejemplos

### Regresión lineal

```python
import tensorc

x = tensorc.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = tensorc.Tensor([[2.0], [4.0], [6.0], [8.0]])

w = tensorc.Tensor([[0.1]], requires_grad=True)
b = tensorc.Tensor([[0.0]], requires_grad=True)
adam = tensorc.Adam([w, b], 0.05)

for _ in range(200):
    loss = x.matmul(w).add(b).mse_loss(y)
    loss.backward()
    adam.step()
```

### Convolución

```python
import tensorc

conv = tensorc.Conv2DLayer(in_channels=3, out_channels=16, kernel_h=3,
                           kernel_w=3, stride=(1, 1), padding=(1, 1),
                           seed=42)
imagen = tensorc.Tensor(datos_nchw)      # [N, 3, H, W]
features = conv.forward(imagen)          # conv + sesgo, con autograd
```

### LSTM sobre una secuencia

```python
import tensorc

lstm = tensorc.LSTMLayer(in_features=8, hidden=16, seed=7)
out = lstm.forward(tensorc.Tensor(datos))   # [seq, batch, in] -> [seq, batch, 16]
```

### Bloque Transformer

```python
import tensorc

bloque = tensorc.TransformerBlock(model_dim=64, ff_dim=128, heads=4, seed=1)
x = tensorc.Tensor(lote).add(tensorc.sinusoidal_positions(seq, 64))  # broadcast
y = bloque.forward(x)                    # [batch, seq, 64]
opt = tensorc.Adam(bloque.params, 0.001) # los 8 parámetros del bloque
```

### Entrenamiento de alto nivel con Model

```python
import tensorc

layer = tensorc.DenseLayer(in_features=784, out_features=10, seed=1)
model = tensorc.Model(
    forward=lambda x: layer.forward(x).softmax(),
    params=[layer.weight, layer.bias],
    loss="cross_entropy",          # o "mse" o callable(pred, target)
    learning_rate=0.01,
)
history = model.fit(x_train, y_train, epochs=50, batch_size=32,
                    shuffle=True)
model.save_weights("modelo.tck")   # checkpoint binario
model2.load_weights("modelo.tck")  # valida formas y copia los pesos
```

### Clasificador estilo scikit-learn

`TensorCClassifier` recibe arrays NumPy y conserva el cálculo de la red,
autograd y Adam en TensorC:

```python
import numpy as np
import tensorc

layer = tensorc.DenseLayer(4, 2, seed=7)
clf = tensorc.TensorCClassifier(
    forward=lambda x: layer.forward(x),
    params=[layer.weight, layer.bias],
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
)
clf.fit(X_train, y_train)
predicciones = clf.predict(X_test)
probabilidades = clf.predict_proba(X_test)
exactitud = clf.score(X_test, y_test)
clf.save("clasificador.tck")
```

La interfaz incluye `fit`, `predict`, `predict_proba`, `score`, `get_params`,
`set_params`, `save` y `load_weights`.

### Guardar y cargar tensores

```python
import tensorc

tensorc.save([w, b], "checkpoint.tck")   # lista de tensores o uno solo
t = tensorc.load("checkpoint.tck")     # lista de Tensors (o uno solo)
t[0].save("uno.tcb")                   # tensor individual
copia = tensorc.load("uno.tcb", requires_grad=True)
```

El repositorio incluye un ejemplo completo de LLM miniature a nivel
carácter (GPT de bolsillo entrenado con el Quijote, atención causal
incluida) en `examples/char_llm_demo.py`.

## Crear tensores

```python
tensorc.Tensor([1, 2, 3, 4], shape=(2, 2))          # desde lista
tensorc.Tensor(numpy_array)                          # desde NumPy (float32/float64)
tensorc.Tensor.zeros((2, 3), requires_grad=True)     # en cero
```

El constructor acepta cualquier objeto con el protocolo de buffer de
Python (NumPy 1.x y 2.x, `array`, memoria contigua): float32 se copia
directo, float64 se convierte, y la shape se infiere del buffer si no
se pasa.

## Correctitud y validación

Los gradientes de todas las operaciones se validan por **diferencias
finitas** en la suite de pruebas del núcleo C (11/11 pruebas en los
backends Accelerate y genérico, sin fugas de memoria detectadas, ASan
limpio), y los bindings se verifican contra referencias NumPy con los
mismos datos y dimensiones.

## Rendimiento

En macOS, matmul (forward y backward), la difusión por filas (sesgos) y
la ruta im2col de convolución usan Apple Accelerate (BLAS); el backend
genérico en C puro sirve para cualquier otra plataforma. Mediciones de
referencia (mismos datos y dimensiones, mediana de varias corridas,
Apple Silicon): inferencia MLP y Conv2D forward **más rápidas que
NumPy** (~1.1× y ~1.4×); entrenamiento de un MLP a ~0.6× de NumPy en el
toy benchmark y ~1.8× más rápido en el dataset real Pima (768 muestras).
Ver `benchmarks/` en GitHub para los scripts reproducibles.

## Compilar desde el código fuente

Requisitos: Python 3.9+ con cabeceras de desarrollo (`python3-dev` en
Linux; el Python de Homebrew en macOS ya las incluye).

```sh
git clone https://github.com/CarlosGuedea/TensorC.git
cd TensorC
python3 -m pip install ./python
```

## Proceso de release (maintainers)

El repositorio incluye un flujo de GitHub Actions
(`.github/workflows/wheels.yml`) que, al crear un tag `v*`, construye el
sdist y las ruedas multiplataforma, ejecuta una prueba de humo en cada
rueda y publica todo en PyPI automáticamente. Para publicar una versión:

```sh
# 1. Actualiza la versión en CMakeLists.txt, include/tensorc/tensorc.h,
#    python/pyproject.toml y python/tensorc/__init__.py
# 2. Ejecuta bash python/sync_core.sh tras cambiar el núcleo C
# 3. Etiqueta y sube:
git tag v0.15.0 && git push origin v0.15.0
```

Para subir a mano (necesitas un token de PyPI):

```sh
cd python
python3 -m pip install build twine
python3 -m build
python3 -m twine upload dist/*
```

## Enlaces

- Repositorio y documentación completa: <https://github.com/CarlosGuedea/TensorC>
- Reporte de issues: <https://github.com/CarlosGuedea/TensorC/issues>

Licencia: Apache-2.0.
