# TensorC — Instrucciones para la máquina con GPU NVIDIA (CUDA)

Este paquete contiene TensorC (framework de redes neuronales en C11 con
autograd) listo para compilar en Linux + NVIDIA. Objetivo en esa máquina:
**medir TensorC CPU contra GPU y empezar el backend CUDA**.

## 1. Requisitos

- Linux x86_64 con NVIDIA GPU y drivers instalados.
- CUDA Toolkit 12.x (`nvcc --version` para verificar).
- CMake ≥ 3.16, gcc o clang.
- Python 3.9–3.13 con NumPy y setuptools (`pip install numpy setuptools`).

## 2. Compilar la biblioteca C y correr la suite (5 min)

```bash
unzip tensorc.zip && cd TensorC
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Deben pasar **11/11 pruebas** (en macOS con Accelerate pasan igual; en
Linux sin Accelerate usa el backend genérico, también deben pasar 11/11).
Ese número es la línea base: si algo falla aquí, no es culpa de la GPU.

## 3. Compilar el paquete Python

```bash
cd python
python3 setup.py build_ext --inplace
cd ..
python3 -c "import sys; sys.path.insert(0,'python'); import tensorc; print(tensorc.__version__)"
```

## 4. Benchmarks reproducibles (mismos datos y dimensiones en ambos lados)

```bash
# Redes neuronales: TensorC vs NumPy (MLP entrenamiento/inferencia, Conv2D)
python3 benchmarks/python_compare/benchmark_vs_numpy.py

# Dataset real: Pima Indians Diabetes (accuracy + tiempos)
python3 benchmarks/python_compare/pima_diabetes.py

# Serie temporal real: predicción diaria del año 2023 (estación Colima)
python3 benchmarks/python_compare/predict_2023_colima.py

# LLM miniature a nivel carácter (Quijote, 2000 pasos ≈ 4–5 min en CPU)
python3 examples/char_llm_demo.py 2000
```

Los tiempos se reportan en milisegundos. Resultados de referencia en
Mac Apple Silicon (CPU): Pima TensorC ~1.8× más rápido que NumPy;
Colima 2023 ~1.26×; LLM ~128 ms/paso con dim 64 × 2 bloques.

## 5. La tarea CUDA (el motivo del viaje)

Arquitectura actual: `ndtensor.c` tiene dos backends detrás de los mismos
ops — genérico (loops C) y Apple Accelerate (BLAS, solo macOS). El
backend CUDA debe ser un **tercer backend tras el mismo switch**, sin
cambiar la API pública:

1. **Primero medir** (un día): corre los benchmarks de la sección 4 en
   la GPU con el código actual (todo CPU) y anota tiempos. Sin línea base
   no hay comparación honesta.
2. **Dispositivo y memoria**: agregar a `tc_tensor_nd` un puntero de
   dispositivo (p. ej. `float *device_data`) + flags, con transferencia
   host↔device perezosa. Empezar con una API `tc_cuda_set_device(int)`.
3. **Ops prioritarios** (en este orden, del perfil medido):
   - `matmul` forward/backward (GEMM vía cuBLAS) — es el 60–70% del
     tiempo del LLM y del MLP.
   - elementwise: add/mul/relu/sigmoid/tanh/softmax (kernels propios).
   - `conv2d` forward/backward (cuDNN o im2col + cuBLAS, espejando la
     ruta Accelerate existente).
   - atención multi-cabeza (el LLM lo compone de primitivas; con las
     anteriores ya acelera sin kernel propio).
4. **Regla del proyecto**: todo cambio se valida con la suite (11/11) y
   diferencias finitas antes de considerarse terminado. No afirmar que
   la GPU es más rápida sin medir contra la sección 4.

## 6. Reglas del proyecto (vigentes en cualquier máquina)

- C11 puro, sin C++. Nombres y comentarios claros.
- No romper la API pública sin justificarlo; no eliminar pruebas.
- Probar todo (ctest 11/11) antes de commitear; evitar fugas de memoria.
- Los releases a PyPI son solo bump de versión + tag `vX.Y.Z` (el CI de
  GitHub Actions construye las ruedas y publica solo).

## 7. Estructura

```
TensorC/
├── ndtensor.c / minitf.c      núcleo C (autograd, ops, capas)
├── include/tensorc/           API pública
├── tests/                       11 pruebas (ctest)
├── examples/                    demos C + char_llm_demo.py (LLM)
├── benchmarks/                  conv2d (C) + comparaciones vs NumPy + datos
├── python/                      paquete pip (bindings CPython)
└── .github/workflows/wheels.yml CI multiplataforma (tags v*)
```
