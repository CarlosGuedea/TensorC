# TensorC

TensorC es un framework de redes neuronales pequeño escrito en C11. Ofrece una API compatible de tensores 2D y modelos densos, además de una API experimental de tensores N-dimensionales con autodiferenciación en CPU. No requiere Python ni dependencias de terceros.

La versión 0.3 está enfocada en CPU y modelos pequeños. Incluye operaciones N-dimensionales, un grafo de operaciones con conteo de referencias y gradientes automáticos para suma, producto elemento a elemento, multiplicación de matrices 2D, ReLU, suma y pérdida MSE. En macOS usa Apple Accelerate para multiplicaciones grandes y conserva un backend genérico para otras plataformas. Todavía no incluye GPU, convoluciones, broadcasting ni importación de modelos de TensorFlow.

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
- Siguientes pasos: broadcasting, serialización, capas convolucionales y backends acelerados.

Las contribuciones son bienvenidas. La API aún puede evolucionar antes de llegar a 1.0.

Licencia Apache-2.0.
