# TensorC

TensorC es un framework de redes neuronales pequeño escrito en C11. Esta primera versión ofrece tensores 2D, suma y multiplicación de matrices, redes densas, activaciones lineal/ReLU/sigmoide/tanh, retropropagación, error cuadrático medio e SGD. No requiere Python ni dependencias de terceros.

La versión 0.1 está enfocada en CPU y en modelos densos pequeños. Aún no incluye autodiferenciación general, GPU, convoluciones ni importación de modelos de TensorFlow.

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

## Usarlo desde otro proyecto CMake

Después de instalar TensorC, agrega el directorio de instalación a `CMAKE_PREFIX_PATH`, enlaza `TensorC::tensorc` e incluye `<tensorc/tensorc.h>`.

## Compilación directa

```sh
cc -O2 -std=c11 -Iinclude -I. minitf.c examples/xor_public_api.c -lm -o tensorc_xor
```

## Estado del proyecto

- 0.1: CPU, tensores 2D, capas densas y SGD.
- 0.2 en desarrollo: acumulación de gradientes por minibatch y Adam.
- Siguientes pasos: serialización, gradientes de operaciones tensoriales y backends acelerados.

Las contribuciones son bienvenidas. La API aún puede evolucionar antes de llegar a 1.0.

Licencia Apache-2.0.
