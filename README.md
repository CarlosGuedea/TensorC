# TensorC

> Manual de instalación, Python, C11, redes neuronales, MNIST, clasificación
> y aprendizaje por refuerzo: [GUIA_USO.md](GUIA_USO.md).

TensorC es un framework de redes neuronales pequeño escrito en C11. Ofrece una API de tensores N-dimensionales con autodiferenciación en CPU. La versión actual es 0.15.0 y se puede instalar desde Python con pip install tensorc.

La versión 0.15.0 está enfocada en CPU y modelos pequeños. Incluye operaciones N-dimensionales, un grafo de operaciones con conteo de referencia, gradientes automáticos para suma, producto elemento a elemento, broadcasting, multiplicación de matrices 2D, convolución 2D con stride y padding configurables, max pooling, reshape dentro del grafo, ReLU, softmax, suma, pérdida MSE y cross-entropy, además de capas densas y convolucionales con inicialización Xavier y optimizador Adam. También permite guardar y cargar checkpoints con un formato binario versionado. En macOS usa Apple Accelerate para multiplicaciones grandes y para la ruta im2col de convoluciones grandes, y conserva un backend genérico para otras plataformas. El ejemplo `tensorc_cnn_mnist` entrena una CNN completa en C puro. Todavía no incluye GPU ni importación directa de modelos de TensorFlow.

El código fuente también incorpora una API C experimental para DQN con acciones discretas, replay buffer y red objetivo. `tc_env` ofrece un contrato de entornos C estilo Gymnasium (`reset`/`step`, `terminated`/`truncated`); `tensorc_dqn_cartpole` muestra entrenamiento sin Python. Consulta [el manual de aprendizaje por refuerzo](GUIA_USO.md#15-aprendizaje-por-refuerzo) para compilarlo y usar el agente Python/NumPy.

## Compilar

Requisitos: CMake 3.16 o superior y un compilador C11.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tensorc_xor
```

Para instalar la biblioteca en una ubicación local:

```sh
cmake --install build --prefix "$HOME/.local"
```

## Ejemplo de API

```c
#include <tensorc/tensorc.h>

size_t layer_sizes[] = {8, 1};
tc_activation activations[] = {TC_SIGMOID, TC_SIGMOID};
tc_model model = tc_model_create(2, layer_sizes, activations, 2);

/* x: una matriz [muestras, 2]; y: [muestras, 1]. */
tc_model_train_epoch(&model, &x, &y, 0.1f);
tc_tensor prediction = tc_model_predict(&model, &sample);

tc_tensor_free(&prediction);
tc_model_free(&model);
```

Consulta [examples/xor_public_api.c](examples/xor_public_api.c) para un programa completo.

## Adam y minibatches

La API incluye SGD y Adam. `batch_size` acumula gradientes antes de actualizar los parámetros:

```c
tc_optimizer optimizer = tc_optimizer_adam(&model, 0.001f);
tc_model_train_epoch_opt(&model, &x, &y, &optimizer, 32);
tc_optimizer_free(&optimizer);
```

El ejemplo `examples/xor_adam.c` usa Adam con un minibatch de cuatro muestras.

## Tensores N-dimensionales y autodiferenciación

La API nueva se incluye desde `<tensorc/tensorc.h>` y usa punteros opacos. Los resultados conservan sus padres mediante conteo de referencias, por lo que se pueden liberar los tensores de entrada después de construir una operación:

```c
const size_t shape[] = {2, 2};
const float values[] = {1, -2, 3, 4};
tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
tc_tensor_nd *y = tc_tensor_nd_relu(x);
tc_tensor_nd *loss = tc_tensor_nd_sum(y);

tc_tensor_nd_backward(loss);
printf("d(loss)/dx[0] = %f\n", tc_tensor_nd_grad(x)[0]);

tc_tensor_nd_free(loss);
tc_tensor_nd_free(y);
tc_tensor_nd_free(x);
```

El ejemplo completo está en [examples/autograd.c](examples/autograd.c) y la prueba de gradientes en [tests/test_autograd.c](tests/test_autograd.c). Para ejecutar las pruebas:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
./build/tensorc_autograd
```

Las operaciones elemento a elemento permiten broadcasting con dimensiones alineadas a la derecha. Una dimensión puede coincidir con la otra o ser igual a uno:

```c
const size_t matrix_shape[] = {2, 3};
const size_t vector_shape[] = {3};
tc_tensor_nd *result = tc_tensor_nd_add(matrix, vector);
```

## Guardar y cargar modelos densos

Los parámetros y la arquitectura de un modelo denso se pueden persistir sin dependencias externas:

```c
tc_model_save(&model, "modelo.tcm");
tc_model loaded = tc_model_load("modelo.tcm");
```

El archivo incluye una firma y una versión de formato para rechazar archivos incompatibles. El formato está diseñado para guardar modelos TensorC en equipos compatibles; no es un formato de intercambio universal entre arquitecturas con distinto endianess o representación de `float`.

## Convolución 2D

La API incluye correlación 2D en CPU con tensores en formato `[lote, canales, alto, ancho]` y kernels `[filtros, canales, alto_kernel, ancho_kernel]`:

```c
tc_tensor_nd *features = tc_tensor_nd_conv2d(input, kernel);

/* Con stride y padding configurables: */
tc_conv2d_options options = {2, 2, 1, 1}; /* stride_h, stride_w, pad_h, pad_w */
tc_tensor_nd *strided = tc_tensor_nd_conv2d_ex(input, kernel, options);
```

`tc_tensor_nd_conv2d` equivale a `tc_tensor_nd_conv2d_ex` con `TC_CONV2D_DEFAULT` (stride 1, sin padding). El padding es simétrico y de ceros; la forma de salida es `[N, K, (H + 2·pad_h − KH)/stride_h + 1, (W + 2·pad_w − KW)/stride_w + 1]`. La operación conserva el grafo de autodiferenciación y calcula gradientes para la entrada y el kernel. El sesgo puede agregarse como un tensor `[1, filtros, 1, 1]` usando broadcasting.

En macOS, las convoluciones grandes usan una ruta im2col + `cblas_sgemm` de Accelerate; los tamaños pequeños y el backend genérico conservan el bucle directo. El benchmark compara contra una implementación equivalente en Python/NumPy con los mismos datos y dimensiones:

```sh
cmake --build build --target tensorc_conv2d_benchmark
python3 benchmarks/conv2d/conv2d_numpy.py
```

En una medición de referencia con entrada `[8,3,32,32]`, kernel `[16,3,3,3]`, stride 1 y padding 1 (medianas: NumPy con 7 repeticiones, C con 200), los resultados fueron:

| Implementación | Forward (mediana) | Backward (mediana) |
|---|---:|---:|
| Python/NumPy (bucles directos) | 321.1 ms | n/d |
| TensorC con Accelerate | 0.384 ms | 0.612 ms |

La salida de C coincidió con NumPy con un error máximo de `4.8e-06`. Los tiempos pueden variar según el equipo.

## Capa convolucional y ejemplo CNN

`tc_conv2d_layer` encapsula kernel y sesgo entrenables con inicialización Xavier determinista. El forward aplica la convolución y suma el sesgo `[1, filtros, 1, 1]` por broadcasting; los gradientes llegan a ambos parámetros y `tc_tensor_nd_sgd_step` aplica el paso de optimización:

```c
tc_conv2d_options options = {1, 1, 1, 1};
tc_conv2d_layer *conv = tc_conv2d_layer_create(1, 8, 3, 3, options, 42);
tc_tensor_nd *features = tc_conv2d_layer_forward(conv, batch); /* conv + sesgo */
/* ... pérdida y backward ... */
tc_tensor_nd_sgd_step(tc_conv2d_layer_kernel(conv), 0.01f);
tc_tensor_nd_sgd_step(tc_conv2d_layer_bias(conv), 0.01f);
tc_conv2d_layer_free(conv);
```

El ejemplo [examples/cnn_mnist.c](examples/cnn_mnist.c) arma una CNN completa —conv(3×3) → ReLU → maxpool(2) → conv(3×3) → ReLU → maxpool(2) → aplanar → matmul + sesgo → softmax → cross-entropy contra one-hot— y la entrena en C puro con Adam:

```sh
./build/tensorc_cnn_mnist            # dataset sintético determinista
./build/tensorc_cnn_mnist train-images.idx3-ubyte train-labels.idx1-ubyte \
                        t10k-images.idx3-ubyte t10k-labels.idx1-ubyte   # MNIST real
```

En una medición de referencia en MacBook Pro M1, el modo sintético (800 imágenes 8×8, 30 épocas) entrenó en ~280 ms con 100 % de precisión en prueba; con MNIST real (8 192 imágenes de entrenamiento, 4 épocas, lotes de 64, Adam lr 0.001) obtuvo 93 % de precisión en 2 048 imágenes de prueba en ~10.1 s de entrenamiento.

### Softmax, cross-entropy y Adam

La API del grafo incluye `tc_tensor_nd_softmax` (sobre la última dimensión) y `tc_tensor_nd_cross_entropy` (media por fila), cuya composición tiene gradiente `(p − t) / filas`. `tc_tensor_nd_optimizer_adam` aplica Adam sobre una lista fija de parámetros con corrección de sesgo:

```c
tc_tensor_nd *params[] = {kernel, bias, weights, out_bias};
tc_tensor_nd_optimizer *opt = tc_tensor_nd_optimizer_adam(params, 4, 0.001f);
/* forward + backward del lote, y después: */
tc_tensor_nd_optimizer_step(opt);
tc_tensor_nd_optimizer_free(opt);
```

## Ejemplo Fahrenheit a Celsius

`examples/fahrenheit.c` y `examples/fahrenheit.py` entrenan la misma red lineal con 2,049 muestras normalizadas, 500 épocas, minibatches de 64, la misma inicialización y la misma tasa de aprendizaje. Ambos imprimen el tiempo de entrenamiento y convierten 100 °F:

```sh
./build/tensorc_fahrenheit
python3 examples/fahrenheit.py
```

El tiempo de C mide el entrenamiento dentro del proceso; el de Python usa `perf_counter()` para el mismo bucle de entrenamiento. En una medición de referencia en el MacBook Pro M1, con siete ejecuciones por programa, la mediana fue:

| Implementación | Mediana | Relación |
|---|---:|---:|
| TensorC en C | 19.963 ms | 1x |
| Python puro | 80.721 ms | 4.0x más lento |

La mejora proviene de reutilizar los buffers internos y evitar copias de cada muestra. Los tiempos pueden variar según el equipo y la versión del compilador.

## Backend Accelerate y benchmark de matrices

En macOS, CMake enlaza automáticamente el framework Apple Accelerate cuando está disponible. Las matrices pequeñas siguen usando el backend genérico para evitar la sobrecarga de una llamada BLAS; las operaciones grandes usan `cblas_sgemm`.

```sh
./build/tensorc_matmul_benchmark
```

Para probar el backend genérico en macOS, configura otra carpeta con `-DTENSORC_USE_ACCELERATE=OFF` y compara ambos ejecutables. En Linux y Windows el backend genérico se mantiene sin dependencias adicionales.

En una medición de referencia con matrices 512×512 y ocho repeticiones, la mediana de siete ejecuciones fue:

| Backend | Mediana | Mejora |
|---|---:|---:|
| Genérico | 80.897 ms | 1x |
| Apple Accelerate | 1.778 ms | 45.5x más rápido |

El benchmark mide tiempo de CPU del proceso y los resultados conservaron el mismo checksum (`0.272000`).

## Uso desde Python (bindings)

El directorio [python/](python/) empaqueta el núcleo C como extensión CPython en
C11, instalable con pip. El núcleo sigue siendo C puro; la extensión solo
convierte objetos:

```sh
python3 -m pip install ./python
```

```python
import tensorc

x = tensorc.Tensor([1, 2, 3, 4], shape=(2, 2), requires_grad=True)
w = tensorc.Tensor([1, 0, 0, 1], shape=(2, 2), requires_grad=True)
loss = x.matmul(w).relu().sum()
loss.backward()
x.sgd_step(0.01)

capa = tensorc.Conv2DLayer(1, 8, 3, 3, stride=(1, 1), padding=(1, 1), seed=42)
opt = tensorc.Adam([capa.kernel, capa.bias, w], 0.001)

# Secuencias: LSTM time-major [seq, batch, in] y atención single-head.
lstm = tensorc.LSTMLayer(in_features=8, hidden=16, seed=7)
out = lstm.forward(tensorc.Tensor(datos))   # [seq, batch, 16]
q = tensorc.Tensor(queries)                  # [batch, seq, depth]
ctx = q.attention(tensorc.Tensor(keys), tensorc.Tensor(values), causal=True)

# Bloque Transformer pre-norm (MHA + residuales + FFN), entrenable.
bloque = tensorc.TransformerBlock(model_dim=64, ff_dim=128, heads=4, seed=1)
x = tensorc.Tensor(lote).add(tensorc.sinusoidal_positions(seq, 64))  # broadcast
y = bloque.forward(x)                        # [batch, seq, 64]
opt = tensorc.Adam(bloque.params, 0.001)     # los 8 parámetros del bloque
```

Para publicar en PyPI: el flujo `.github/workflows/wheels.yml` construye
y sube automáticamente las ruedas multiplataforma (macOS x86_64/arm64,
Linux x86_64/aarch64, Windows, CPython 3.9–3.13) y el sdist al crear un
tag `v*`; en local, `python3 -m build` dentro de `python/` y
`python3 -m twine upload dist/*` también funcionan. Para ruedas
multiplataforma en local usa [cibuildwheel](https://cibuildwheel.readthedocs.io/).
El núcleo C se vendoriza en `python/tensorc/core/` (actualízalo con
`python/sync_core.sh`).

## Usarlo desde otro proyecto CMake

Después de instalar TensorC, agrega el directorio de instalación a `CMAKE_PREFIX_PATH`, enlaza `TensorC::tensorc` e incluye `<tensorc/tensorc.h>`.

## Compilación directa

```sh
cc -O2 -std=c11 -Iinclude -I. minitf.c examples/xor_public_api.c -lm -o tensorc_xor
```

## Estado del proyecto

- 0.1: CPU, tensores 2D, capas densas y SGD.
- 0.2: acumulación de gradientes por minibatch y Adam.
- 0.3: tensores N-dimensionales, autodiferenciación, entrenamiento CPU optimizado y backend Accelerate para matmul.
- 0.4: broadcasting en operaciones elemento a elemento y acumulación correcta de gradientes.
- 0.5: serialización binaria versionada de modelos densos.
- 0.6: convolución 2D válida con gradientes para entrada y kernels.
- 0.7: stride y padding configurables en Conv2D, ruta im2col+GEMM para convoluciones grandes en Accelerate, y benchmark reproducible contra NumPy.
- 0.8: capa convolucional de alto nivel, max pooling y reshape en el grafo, pasos SGD, y ejemplo CNN entrenable en C puro con MNIST (IDX) o dataset sintético.
- 0.9: softmax y cross-entropy en el grafo, optimizador Adam sobre listas de parámetros; la CNN de ejemplo llega a 93 % en MNIST.
- 0.10: bindings oficiales de Python: extensión CPython escrita en C11 sobre el mismo núcleo, instalable con `pip install ./python` y publicable en PyPI.
- 0.11: activaciones sigmoid y tanh en el grafo N-dim con autograd, capa densa de alto nivel (`tc_dense_layer` / `DenseLayer`) con inicialización Xavier determinista.
- 0.12: LSTM de alto nivel sobre secuencias completas [seq, batch, in], atención single-head con máscara causal opcional, y primitivas de secuencia en el grafo (slice en eje, concat N-aria, layer norm y transposición); refactor interno a N padres por nodo y fix de fuga en el forward LSTM. Expuestos a Python: `LSTMLayer` y los métodos `slice_axis`, `concat`, `layer_norm`, `transpose` y `attention`.
- 0.13: bloque Transformer completo: operación `permute` (permutación general de ejes con autograd), atención multi-cabeza, codificación posicional sinusoidal y capa `tc_transformer_block` / `TransformerBlock` pre-norm (MHA + residuales + FFN) entrenable end-to-end con Adam.
- Siguientes pasos: pila de bloques + embeddings para un encoder completo, guardado/carga de la CNN y GPU (Metal en macOS / CUDA en Linux para comparativas de tiempos).

Las contribuciones son bienvenidas. La API aún puede evolucionar antes de llegar a 1.0.

Licencia Apache-2.0.
