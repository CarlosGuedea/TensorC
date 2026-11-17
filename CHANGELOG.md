# Changelog

## 0.4.0 — broadcasting en tensores N-dimensionales

- `add` y `mul` aceptan dimensiones compatibles alineadas a la derecha.
- Los gradientes se acumulan correctamente en los ejes transmitidos.
- Se agregó una prueba CTest para resultados y gradientes con matrices y vectores.

## 0.3.2 — backend Accelerate para álgebra lineal

- Uso opcional de Apple Accelerate/BLAS para multiplicaciones grandes de matrices.
- Backend genérico conservado para macOS sin Accelerate, Linux y Windows.
- Umbral para evitar la sobrecarga BLAS en matrices pequeñas.
- Benchmark `tensorc_matmul_benchmark` y prueba de exactitud `tensorc_matmul`.
- En la prueba 512×512, Accelerate obtuvo una mediana de 1.778 ms frente a 80.897 ms del backend genérico.

## 0.3.1 — optimización del entrenamiento CPU

- Reutilización de buffers de activaciones, valores intermedios y deltas durante el entrenamiento.
- Eliminación de copias temporales de cada fila del dataset mediante vistas de memoria.
- El benchmark Fahrenheit→Celsius pasó de aproximadamente 169.77 ms a 19.96 ms.
- En la misma prueba, C fue aproximadamente 4.0 veces más rápido que Python puro.

## 0.3.0 — tensores N-dimensionales y autodiferenciación

- API pública `tc_tensor_nd` para tensores N-dimensionales en formato row-major.
- Creación desde datos, reshape, acceso a forma, datos y gradientes.
- Operaciones `add`, producto elemento a elemento, `matmul` 2D, ReLU, suma y MSE.
- Grafo de operaciones con conteo de referencias y autodiferenciación hacia atrás.
- Ejemplo `tensorc_autograd` y prueba CTest de gradientes.
- Ejemplo equivalente C/Python de conversión Fahrenheit a Celsius con minibatches.
- Se conserva la API 2D anterior (`tc_tensor`, modelos densos, SGD y Adam).

## 0.1.0 — primera versión pública en preparación

- Biblioteca C11 para tensores 2D y operaciones básicas de matrices.
- Redes densas con activaciones lineal, ReLU, sigmoid y tanh.
- Retropropagación, MSE y entrenamiento SGD muestra por muestra.
- API pública C, compilación e instalación con CMake.
- Ejemplo XOR.

## 0.2.0 — optimización y minibatches

- Optimizador Adam con corrección de sesgo.
- Acumulación de gradientes y entrenamiento por minibatches.
- Ejemplo XOR con Adam.
