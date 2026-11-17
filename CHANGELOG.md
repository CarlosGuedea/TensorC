# Changelog

## 0.8.0 — capa convolucional y ejemplo CNN

- Nueva operación de grafo `tc_tensor_nd_maxpool2d` (ventanas no traslapadas) con gradiente hacia el máximo de cada ventana.
- Nueva operación de grafo `tc_tensor_nd_reshape_to` para aplanar tensores dentro del grafo de autodiferenciación.
- `tc_conv2d_layer`: kernel y sesgo entrenables con inicialización Xavier determinista; el forward aplica convolución + sesgo con broadcasting.
- `tc_tensor_nd_sgd_step` aplica un paso de SGD sobre cualquier tensor con gradiente.
- Ejemplo `tensorc_cnn_mnist`: CNN completa (conv → ReLU → maxpool → … → matmul + sesgo → MSE) entrenable en C puro; carga MNIST en formato IDX cuando se le pasan los archivos y usa un dataset sintético determinista como respaldo.
- Pruebas de valores, gradientes por diferencias finitas y entrenamiento end-to-end de la capa.

## 0.7.0 — Conv2D con stride y padding

- Se agregó `tc_tensor_nd_conv2d_ex` con opciones de stride y padding simétrico de ceros (`tc_conv2d_options`); `tc_tensor_nd_conv2d` se conserva como stride 1 / sin padding.
- Los gradientes automáticos de entrada y kernel respetan stride y padding.
- Validación ampliada: stride en cero, kernel mayor que la entrada aun con padding, y desbordamiento en las dimensiones rellenadas.
- Ruta im2col + GEMM (Accelerate) para convoluciones grandes, con umbral para conservar el bucle directo en tamaños pequeños y en el backend genérico.
- Pruebas de valores conocidos con stride/padding, identidad de composición con varios canales y filtros, y diferencias finitas con tolerancia relativa al ruido de float32.
- Benchmark reproducible `tensorc_conv2d_benchmark` + `benchmarks/conv2d/conv2d_numpy.py` con los mismos datos y dimensiones: en `[8,3,32,32]` × `[16,3,3,3]`, mediana de 0.384 ms (forward) frente a 321.1 ms de NumPy, con salida idéntica dentro de 4.8e-06.

## 0.6.0 — convolución 2D en CPU

- Se agregó `tc_tensor_nd_conv2d` para entrada `[N,C,H,W]` y kernels `[K,C,KH,KW]`.
- La operación calcula gradientes automáticos para la entrada y los kernels.
- Se agregó una prueba CTest de valores de salida y gradientes.

## 0.5.0 — serialización de modelos densos

- Se agregaron `tc_model_save` y `tc_model_load`.
- El formato binario incluye firma, versión, arquitectura, activaciones, pesos y sesgos.
- Se rechazan archivos incompletos, incompatibles o con dimensiones inválidas.
- Se agregó una prueba CTest de guardado, carga y equivalencia de predicciones.

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
