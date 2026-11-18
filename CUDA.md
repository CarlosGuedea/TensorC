# Backend CUDA de TensorC

Tercer backend de cómputo tras el mismo switch que el genérico (loops C) y
Apple Accelerate: cada operación decide por tamaño si corre en GPU y cae al
camino genérico ante cualquier fallo de infraestructura (sin GPU, sin
memoria de device). La API pública no cambia: el mismo código y las mismas
pruebas corren en CPU o GPU según cómo se compiló la biblioteca.

Estado: **v3 validada en la máquina con GPU** (RTX A2000 Laptop GPU,
sm_86, bajo WSL2): suite 11/11 normal y 11/11 con `TC_CUDA_MIN_ELEMS=0`,
y benchmarks medidos contra la línea base CPU (ver "Resultados medidos"
más abajo). La v2 añadió atención, `layer_norm`, permute/transpose y
reshape; la v3 añade `cross_entropy`, SGD/Adam y `zero_grad` en device,
un **pool de memoria de device** y **sincronía diferida** (los wrappers
no llaman `cudaDeviceSynchronize`: todo se encola en el stream por
defecto y el host solo sincroniza cuando lee datos).

## Qué corre en GPU

| Operación | Forward | Backward | Vía |
|---|---|---|---|
| `matmul` (m·n·k ≥ 4096) | ✔ | ✔ | cuBLAS (GEMM row-major reordenado) |
| `add` / `mul` con broadcasting | ✔ | ✔ | kernels propios |
| `relu` / `sigmoid` / `tanh` | ✔ | ✔ | kernels propios |
| `softmax` (última dimensión) | ✔ | ✔ | kernel propio (un hilo por fila) |
| `layer_norm` (última dimensión) | ✔ | ✔ | kernel propio (un hilo por fila) |
| `attention` (single y multi-cabeza) | ✔ | ✔ | GEMM batidas cuBLAS + softmax causal propio |
| `permute` / `transpose` | ✔ | ✔ | kernel propio (biyección por índices) |
| `reshape` | ✔ | ✔ | copia device→device / acumulación |
| `cross_entropy` | ✔ | ✔ | reducción a dos niveles (bloque + atómico) |
| `sgd_step` / `optimizer_step` (Adam) | — | ✔ | kernel in-place; momentos m/v viven en device |
| `zero_grad` | — | ✔ | memset en device (no baja ceros al host) |

El resto (`conv2d`, pooling, `slice`, `concat`, otras reducciones) sigue
en CPU en esta v3.

**Regla de residencia**: además del umbral de tamaño (16 384 elementos por
defecto), una op va a GPU si su entrada ya reside en device — caer a CPU en
medio de una cadena pagaría el par de copias D2H+H2D por puro traslado. Es
lo que mantiene el bloque Transformer completo en device: sus tensores
intermedios (p. ej. 196 K elementos con dim 64, seq 64, batch 48) y las
ops pequeñas entre GEMM nunca bajan al host.

## Modelo de memoria perezoso

Cada `tc_tensor_nd` gana (solo compilando con CUDA) dos punteros de device
(`dev_data`, `dev_grad`) y cuatro flags de validez. Invariante: **al menos
una copia (host o device) está al día en todo momento**.

- Una op con ruta CUDA deja su resultado en device y marca la copia host
  como obsoleta; las ops encadenadas leen directamente de device sin
  transferencias intermedias.
- La bajada D2H ocurre perezosamente: accesores (`tc_tensor_nd_data`,
  `const_data`, `grad`), rutas genéricas de ops sin kernel y
  serialización sincronizan antes de leer/escribir en host.
- `tc_cuda_set_device(int)` elige el dispositivo (v1: uno por proceso).
- **Sincronía diferida**: los wrappers NO llaman `cudaDeviceSynchronize`;
  kernels, GEMMs, memsets y copias se encolan en el stream por defecto
  (orden garantizado entre ops) y el host solo sincroniza en las copias
  D2H. Los wrappers comprueban el error de lanzamiento; un fallo de
  ejecución se detecta en el siguiente punto de sincronía y aborta con
  diagnóstico (política de dispositivo irrecuperable, detección
  diferida). Esto eliminó ~1 ms de overhead por op bajo WSL2.
- **Pool de memoria de device**: `tc__cuda_alloc/free` reutilizan bloques
  por tamaño exacto en vez de llamar al driver en cada tensor
  (cudaMalloc/cudaFree sincronizan el dispositivo). Seguro porque todo va
  al mismo stream; los bloques se liberan al cerrar el proceso.
- Fallos de reserva o subida de memoria → la op cae a CPU. Un fallo de
  dispositivo a mitad de un backward (GPU muerta) aborta con diagnóstico:
  la alternativa sería devolver gradientes corruptos en silencio.
- **Puntero host expuesto (`host_data_exposed`)**: los accesores
  `tc_tensor_nd_data()` / `tc_tensor_nd_grad()` devuelven el puntero crudo
  de host y nada impide que el llamador lo siga mutando después (las
  pruebas lo hacen). Por eso, una vez expuesto el puntero, la subida H2D
  se repite en cada op CUDA en lugar de confiar en una invalidación
  única: sin este flag, una resubida con datos viejos revalidaba el
  device y las diferencias finitas daban 0 exacto. El coste es una copia
  extra solo para tensores cuyo puntero fue entregado al usuario.

## Compilar (biblioteca C)

Requisitos: CUDA Toolkit 12.x, CMake ≥ 3.18 (por `find_package(CUDAToolkit)`).

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DTENSORC_USE_CUDA=ON
cmake --build build-cuda -j
```

## Compilar (paquete Python con GPU)

```bash
cd python
TENSORC_CUDA=1 python3 setup.py build_ext --inplace
cd ..
python3 -c "import sys; sys.path.insert(0,'python'); import tensorc; \
print(tensorc.__version__, tensorc.cuda_available())"
```

`setup.py` localiza el toolkit con `CUDA_HOME`/`CUDA_PATH` (por defecto
`/usr/local/cuda`). Los benchmarks de la sección 4 de INSTRUCCIONES.md
(incluido `char_llm_demo.py`) aceleran sin tocar una línea: las ops eligen
GPU por tamaño internamente. Las ruedas de PyPI se publican sin CUDA; el
build con GPU es para la máquina de benchmarks.

**Ojo con la arquitectura**: `setup.py` compila con `nvcc -arch=native`
(anulable con `TENSORC_CUDA_ARCH="86"`). Es obligatorio: el default de
nvcc genera SASS sm_52, y en GPUs más nuevas bajo WSL2 el driver no
devuelve un error manejable sino que **crashea el proceso en el primer
`cudaLaunchKernel`** (segfault sin diagnóstico). El build CMake usa
`CMAKE_CUDA_ARCHITECTURES=86` por la misma razón.

## Validación (checklist en la máquina con GPU)

Regla del proyecto: nada se da por terminado sin la suite y sin medir.

1. **Línea base CPU primero** (mismo equipo, build sin CUDA):
   `ctest` 11/11 + benchmarks de INSTRUCCIONES.md sección 4; anotar tiempos.
2. **Suite con las rutas CUDA forzadas** — las pruebas usan tensores
   pequeños, así que se baja el umbral a cero para que TODO pase por la GPU
   (incluye las GEMM de cuBLAS) y las diferencias finitas las validen:
   ```bash
   TC_CUDA_MIN_ELEMS=0 ctest --test-dir build-cuda --output-on-failure
   ```
   Deben pasar **11/11**. Ojo: los backward de `add`/`mul` usan `atomicAdd`
   (orden no determinista, diferencias ~1e-7); las tolerancias de las
   pruebas lo absorben.
3. **Fugas y errores de device**: `compute-sanitizer --tool memcheck
   ./build-cuda/tensorc_test_autograd` (y el resto de la suite si hay
   dudas).
4. **Benchmarks GPU vs la línea base del paso 1** (misma máquina):
   `./build-cuda/tensorc_matmul_benchmark`,
   `python3 benchmarks/python_compare/benchmark_vs_numpy.py`,
   `python3 examples/char_llm_demo.py 2000` con el paquete compilado con
   `TENSORC_CUDA=1`. Reportar tiempos de CPU y GPU lado a lado.

## Resultados medidos (2026-11-17/18)

Máquina: NVIDIA RTX A2000 Laptop GPU (sm_86) bajo WSL2 Ubuntu, CUDA
12.9.86 (toolkit vía micromamba), gcc 13.3.0. CPU de línea base: Xeon
E5-2697 v3 (2014, 14c/28t, muy lenta en monohilo — contexto para no
sobreestimar los factores).

- **Suite**: 11/11 en build normal y 11/11 con `TC_CUDA_MIN_ELEMS=0`
  (todas las rutas CUDA forzadas, validadas por diferencias finitas).
- **matmul 512×512 vía API de grafo** (`nd_matmul_bench.c`, media con
  warmup): CPU genérico 30.767 ms → GPU 0.914 ms en la v1 (sync por op),
  **0.339 ms con la v3 (sincronía diferida), ~91×**, checksum idéntico
  (0.335800). GPU limitada por la latencia de lanzamiento, no por cuBLAS
  puro.
- **LLM** (`examples/char_llm_demo.py`, dim 64, 2 bloques, 2 cabezas,
  seq 64, batch 48): CPU 2047 ms/paso → **153 ms/paso con la v3**
  (**~13.4×**); la loss converge igual (3.355→2.372 en 300 pasos GPU).
  Desglose por versión: v1 (matmul/elementwise) 658 ms; v2 (+atención,
  layer_norm, permute, reshape) 309 ms; v3 (+cross_entropy, Adam,
  zero_grad, pool de memoria y sincronía diferida) 153 ms.

### Diagnóstico del tiempo restante (v3)

Medido con micro-benchmarks sobre tensores ya residentes:

- **~0.8-0.9 ms por op de puro lanzamiento**: una ReLU sobre 196 K
  elementos (cómputo ~8 µs) tarda 0.84 ms de media encadenada; un matmul
  [3072,64]×[64,64] (cómputo ~10 µs) 0.86 ms. El paso del LLM lanza
  ~300 ops (incluye los memsets de `zero_grad` por nodo), así que el
  tiempo restante es casi todo latencia de lanzamiento WSL2/WDDM, no
  cómputo.
- El lado host no cuesta nada: la misma cadena por el camino CPU con
  tensores 4×4 (Python + binding + grafo) mide 0.001 ms/op.
- En Linux nativo el lanzamiento cuesta ~10 µs: el mismo código debería
  bajar el paso a un dígito de ms. La palanca definitiva bajo WSL2 es
  **CUDA Graphs** (capturar el paso y relanzarlo como un solo grafo),
  que exige buffers estáticos: cambio de arquitectura mayor.

Los pasos 3 (compute-sanitizer) y los benchmarks restantes del paso 4
(`benchmark_vs_numpy.py`, CNN MNIST) quedan pendientes; la suite 11/11 y
las dos mediciones de arriba son las que respaldan el estado "validado".

## Limitaciones conocidas de la v3

- Un solo dispositivo por proceso; un solo stream (el por defecto).
  Suficiente con la sincronía diferida, porque el cuello medido es la
  latencia de lanzamiento WSL2 (~0.8 ms/op), no la falta de solape.
- La siguiente palanca de rendimiento es **CUDA Graphs** (ver
  "Diagnóstico del tiempo restante"): requiere que el paso opere sobre
  buffers estáticos para poder capturarlo.
- Softmax, `layer_norm` y el dS de atención usan un hilo por fila; si el
  perfil lo pide (filas anchas), el siguiente paso es un bloque por fila
  con reducciones en memoria compartida.
- La atención materializa scores (scratch P y dP de `batch·q_seq·k_seq`
  floats por llamada, salen del pool); una variante estilo
  flash-attention lo evitaría — solo si el perfil lo justifica.
- `conv2d`, pooling, `slice` y `concat` siguen en CPU (no están en el
  camino caliente del LLM; portarlos es mecánico si un perfil lo pide).
- Las transferencias D2H perezosas convierten ops encadenadas en casi
  libres, pero una alternancia GPU→CPU→GPU por ops sin kernel paga dos
  copias; la regla de residencia minimiza estos cruces.

## Archivos

- `tensorc_cuda.cu` — backend: dispositivo, memoria, kernels, cuBLAS.
- `tensorc_cuda_internal.h` — interfaz interna C11 entre `ndtensor.c` y el
  backend (buffers crudos + descriptor de broadcasting).
- `ndtensor.c` — campos de device en el tensor, sincronía perezosa, rutas
  rápidas y dispatcher de backward tras `#ifdef TENSORC_HAS_CUDA`.
- `include/tensorc/ndtensor.h` — API pública `tc_cuda_available /
  tc_cuda_set_device / tc_cuda_device` (con stubs sin el backend).
- `python/` — build opt-in con `TENSORC_CUDA=1` y funciones
  `tensorc.cuda_available()`, `.cuda_set_device(i)`, `.cuda_device()`.
