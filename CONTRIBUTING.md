# Contribuir a TensorC

Gracias por ayudar a mejorar TensorC. La API está en etapa 0.x y puede cambiar antes de 1.0.

## Preparar el entorno

Se requiere un compilador C11 y CMake 3.16 o superior.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Antes de enviar una contribución, compila la biblioteca y ejecuta el ejemplo XOR:

```sh
./build/tensorc_xor
```

## Cambios de API

- Mantén la interfaz pública bajo `include/tensorc/`.
- Documenta cambios visibles para usuarios en `README.md` y `CHANGELOG.md`.
- Evita dependencias nuevas sin explicar su necesidad y compatibilidad.
- Incluye ejemplos pequeños para funciones nuevas.
- No cambies silenciosamente el formato de modelos una vez que se publique.

## Pull requests

Describe el problema, la solución y las plataformas donde compilaste. Para cambios numéricos, incluye las entradas y salidas esperadas y cómo verificarlas.
