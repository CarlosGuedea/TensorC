# Changelog

## 0.16.0 — backend Metal opcional

- Nuevo backend opcional Apple Metal para operaciones de tensores en macOS; las operaciones no soportadas siguen en CPU y `tensorc.metal_available()` indica si se activó.
- CMake y la extensión CPython pueden compilar Metal; las ruedas Apple Silicon de PyPI se construyen con este backend, mientras Intel macOS, Linux y Windows conservan sus rutas CPU/CUDA.
- La medición inicial de MNIST no mostró una aceleración de entrenamiento significativa frente a CPU; no se prometen mejoras de rendimiento hasta optimizar esa ruta.
- Flujo de publicación PyPI preparado para Trusted Publishing y sin token API persistente en GitHub Actions.

## 0.14.0 — Model de alto nivel, serialización, avgpool y optimizaciones

- Nueva `tensorc.Model`: envuelve un forward + lista de parámetros con `fit` (mini-batches con shuffle), `evaluate`, `predict`, `save_weights` y `load_weights`. Python puro sobre el núcleo C.
- Nueva operación `avgpool2d` (avg pooling 2D no traslapado) con backward que reparte el gradiente entre la ventana; binding `Tensor.avgpool2d` y pruebas con valores conocidos, gradiente analítico y diferencias finitas.
- Serialización N-D: `tc_tensor_nd_save/load` (tensor único) y `tc_save_checkpoint` / `tc_load_checkpoint` / `tc_free_checkpoint` (lista de parámetros en un archivo binario versionado, validación estricta de formato). En Python: `tensorc.save`, `tensorc.load`, `Tensor.save` y el primitivo `tc_tensor_nd_copy_` (`Tensor.copy_`, copia de datos con validación de forma).
- Optimizaciones en el núcleo (motivadas por benchmarks reproducibles contra NumPy con mismos datos/dimensiones): matmul forward y backward vía GEMM de Accelerate (el entrenamiento de un MLP 784→128→10 pasó de 4156 ms a 46.6 ms) y ruta rápida para la difusión por filas `[rows,cols]` contra `[1,cols]` en add/mul (forward y backward).
- Nuevos benchmarks reproducibles en `benchmarks/python_compare/`: comparación MLP/Conv2D vs NumPy; Pima Indians Diabetes (TensorC ~1.8× más rápido, accuracy equivalente); predicción diaria del año 2023 de la estación Colima 6040 (TensorC ~1.26× más rápido, MSE equivalente).
- Nuevo ejemplo `examples/char_llm_demo.py`: LLM miniature a nivel carácter (GPT de bolsillo) entrenado con el Quijote — embeddings one-hot, posiciones sinusoidales, bloques pre-norm con atención causal, softmax + cross-entropy, Adam y muestreo con temperatura. Loss 3.36 → 1.78 en 2000 pasos (255 s).
- Suite: 11/11 en Accelerate y 11/11 en el backend genérico, ASan limpio y 0 fugas.

## 0.13.1 — documentación del paquete

- README del paquete Python (`python/README.md`) reescrito por completo: instalación con `pip install tensorc`, features (regresión, Conv2D, LSTM, atención, bloque Transformer), ejemplos de uso, creación de tensores, validación de gradientes, rendimiento, build desde fuente y proceso de release con cibuildwheel.
- Sin cambios de código. El único propósito de esta release es publicar la documentación actualizada en PyPI y validar de punta a punta el flujo de ruedas multiplataforma (macOS × 2, Windows, Linux × 2, CPython 3.9–3.13).

## 0.13.0 — bloque Transformer: atención multi-cabeza y permute

- Nueva operación de grafo `tc_tensor_nd_permute`: permutación general de ejes (`out[i] = in[axes[i]]` por dimensión), con gradiente por el mapeo inverso validado por diferencias finitas.
- Nueva `tc_tensor_nd_multihead_attention`: atención multi-cabeza sobre [batch, seq, depth] componiendo permute + reshape + atención single-head + concat, así el autograd fluye sin nodos nuevos; soporta máscara causal. Validada contra referencia independiente y diferencias finitas.
- Nueva codificación posicional `tc_tensor_nd_sinusoidal_positions(seq, depth)` (hoja constante, seno en pares y coseno en impares).
- Nueva capa de alto nivel `tc_transformer_block`: bloque pre-norm (x + MHA(LN(x)) y x + FFN(ReLU)(LN(x))) sobre [batch, seq, model_dim], con 8 parámetros entrenables (W_q/W_k/W_v/W_o, W1/b1, W2/b2) en Xavier determinista; `tc_transformer_block_param[i]` expone la lista para Adam.
- El bloque aprende end-to-end: en la prueba reduce la pérdida del objetivo "promedio de la secuencia" a menos de la mitad con Adam.
- Bindings de Python: métodos `permute` y `multihead_attention` en `Tensor`, clase `TransformerBlock` (prop `params` para Adam) y función `sinusoidal_positions`.
- Suite: 10/10 en Accelerate y 10/10 en el backend genérico, ASan limpio y 0 fugas en los 10 tests.

## 0.12.0 — LSTM, atención y primitivas de secuencia

- Nueva capa de alto nivel `tc_lstm_layer`: LSTM sobre una secuencia completa [seq, batch, in] (time-major) → [seq, batch, hidden], con pesos W_ih [4H, in], W_hh [4H, H] y sesgo [1, 4H] compartidos entre pasos (los gradientes se acumulan). El forward despliega el grafo paso a paso con slice/concat, así la salida retiene todo el historial para backward.
- Nuevas operaciones de grafo: `tc_tensor_nd_slice_axis` (slice contiguo en un eje), `tc_tensor_nd_concat` (N tensores en un eje), `tc_tensor_nd_layer_norm` (sobre la última dimensión, sin afín), `tc_tensor_nd_transpose` (rango 2) y `tc_tensor_nd_attention` (single-head, softmax(QKᵀ/√d + máscara)V sobre [batch, seq, depth], con máscara causal opcional; recomputa P en backward sin retenerlo).
- Refactor interno del grafo: los padres de cada nodo pasan de campos fijos `a`/`b` a un arreglo dinámico `parents` de tamaño arbitrario (necesario para concat de N y attention de 3 padres); la API pública existente no cambia.
- Fix de fuga de memoria en el forward LSTM: las slices de las puertas se crean en variable local y se liberan tras crear su activación (la referencia de creación debe soltarse explícitamente); `leaks` reporta 0 bytes en toda la suite.
- Validación: forward LSTM contra implementación de referencia independiente, gradientes por diferencias finitas y entrenamiento end-to-end (copy-task con Adam, la pérdida se reduce a menos de la mitad).
- Bindings de Python: métodos `slice_axis`, `concat`, `layer_norm`, `transpose` y `attention` en `Tensor`, y nueva clase `LSTMLayer` (props `weight_ih`/`weight_hh`/`bias`, método `forward`).
- Suite: 9/9 en Accelerate y 9/9 en el backend genérico, ASan limpio y 0 fugas en los 9 tests; MNIST al 100 % en ambos backends.

## 0.11.0 — sigmoid, tanh y capa densa

- Nuevas operaciones de grafo `tc_tensor_nd_sigmoid` (rama estable según el signo, sin NaN/inf en entradas extremas) y `tc_tensor_nd_tanh`, ambas con gradiente analítico validado por diferencias finitas.
- Nueva capa de alto nivel `tc_dense_layer`: peso [in, out] y sesgo [1, out] entrenables, Xavier uniforme determinista; el forward hace matmul + sesgo con broadcasting sobre la dimensión de batch.
- Rechazo de formas inválidas: la entrada del forward debe ser rango 2 con in_features columnas.
- Bindings de Python: métodos `sigmoid`/`tanh` en `Tensor` y nueva clase `DenseLayer` (props `weight`/`bias`, método `forward`).
- Nueva prueba `tensorc_dense_layer`: valores conocidos, gradientes analíticos de entrada/peso/sesgo, estabilidad numérica de sigmoid, rechazos y entrenamiento end-to-end (la capa aprende y = 2x − 1 por SGD).
- Suite: 8/8 en Accelerate y 8/8 en el backend genérico.

## 0.10.0 — bindings de Python (paquete `tensorc` en pip)

- Nuevo módulo de extensión CPython escrito en C11 (`python/tensorc/_tensorc.c`) sobre el mismo núcleo del repo: sin C++, sin ctypes, sin dependencias externas más allá de CPython y NumPy (solo como formato de intercambio).
- Expone `Tensor` (constructor desde datos planos + shape, `zeros`, `backward`, `sgd_step`, `add`, `mul`, `matmul`, `relu`, `softmax`, `sum`, `mse_loss`, `cross_entropy`, `maxpool2d`, `reshape`, `conv2d` con stride/padding), `Conv2DLayer` y `Adam`.
- El constructor `Tensor(data, ...)` acepta cualquier objeto con protocolo de buffer (NumPy 1.x y 2.x, `array`, memoria contigua): float32 se copia directo, float64 se convierte, y la shape se infiere del buffer si no se pasa.
- Empaquetado con `pyproject.toml` + `setup.py`: `pip install ./python` compila el núcleo C y la extensión en una sola pasada; publicable en PyPI con `python -m build` + `twine` (ver `python/README.md`).
- El núcleo se vendoriza en `python/tensorc/core/` mediante `python/sync_core.sh` para que el paquete sea autocontenido; los cambios al núcleo C requieren re-sincronizar antes de publicar.
- Verificado: conv2d con valores y gradientes exactos contra referencia, entrenamiento softmax+CE+Adam al 95 %, estrés de 10 000 grafos sin fugas ni crashes. La suite C sigue intacta (7/7 en ambos backends).

## 0.9.0 — softmax, cross-entropy y Adam en el grafo

- Nueva operación `tc_tensor_nd_softmax` (última dimensión) con gradiente estable.
- Nueva pérdida `tc_tensor_nd_cross_entropy` (media por fila, probabilidades clamp a 1e-7).
- Nuevo `tc_tensor_nd_optimizer_adam`: Adam sobre una lista fija de parámetros del grafo N-dim (betas 0.9/0.999, eps 1e-8, con corrección de sesgo).
- El ejemplo `tensorc_cnn_mnist` usa softmax + cross-entropy + Adam: MNIST sube de 71 % a 93 % en las mismas 4 épocas; el sintético de 80 % a 100 %.
- Pruebas de valores, gradientes por diferencias finitas (incluida la identidad d(softmax+CE)/dx = (p − t)/filas) y convergencia de Adam.

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
