# Clasificador de lugares para estudiar con TensorC

Experimento realizado con la versión actual de TensorC (0.14.0) y el ZIP
`Datos-Supabase-2026-11-18.zip`.

## Modelo

La red es multimodal y está escrita en C11 sobre la API de TensorC:

- Rama visual: `Conv2D(3→8)` → ReLU → MaxPool → `Conv2D(8→16)` → ReLU → MaxPool.
- Rama de registro: 11 variables normalizadas del dataset.
- Fusión de ambas ramas y clasificador `Dense(24→2)`.
- Optimizador Adam, 35 épocas y muestreo balanceado entre `apto` y `no_apto`.

Las 43 muestras `incierto` no se usaron para entrenar. De las 303 muestras
etiquetadas, la partición fue 211 entrenamiento, 45 validación y 47 prueba.

## Resultado de prueba

| Métrica | Resultado |
|---|---:|
| Exactitud | 78.72 % |
| Precisión de `apto` | 84.62 % |
| Recall de `apto` | 57.89 % |
| F1 de `apto` | 68.75 % |
| ROC-AUC aproximado | 0.8853 |
| Matriz de confusión | `[[26, 2], [8, 11]]` |

El entrenamiento tomó aproximadamente 4.4 segundos en el equipo usado. El
checkpoint ocupa aproximadamente 70 KB.

## Archivos

- `study_classifier.c`: entrenamiento e inferencia en C11/TensorC.
- `train_study_classifier.py`: entrenamiento desde Python usando los bindings de TensorC.
- `train_study_classifier.ipynb`: versión para Jupyter usando el dataset descomprimido en una carpeta raíz.
- `results/study_classifier_multimodal.tck`: pesos entrenados.
- `results/test_predictions_multimodal.csv`: predicciones de prueba.
- `results/uncertain_predictions_multimodal.csv`: clasificación de las 35 fotos inciertas disponibles.
- `results/manifest.csv`: relación entre índices, etiquetas y fotografías.

La ejecución desde Python generó un segundo checkpoint en
`results/python/study_classifier_python.tck` y obtuvo 76.60 % de exactitud,
71.79 % de F1 para `apto` y ROC-AUC de 0.8515.

## Ejecución desde Python

```bash
PYTHONPATH=/ruta/a/tensorc/python python3 train_study_classifier.py \
  --zip Datos-Supabase-2026-11-18.zip \
  --output results/python
```

En Jupyter, descomprime el ZIP y configura `DATASET_DIR` apuntando a la carpeta
que contiene `datos.csv` y `fotos/`.

## Interpretación

El resultado es una prueba de concepto con un dataset pequeño. Las 35 muestras
inciertas se clasificaron, pero 19 quedaron con probabilidad intermedia
(`0.35 ≤ P(apto) ≤ 0.65`), por lo que conviene revisarlas manualmente. Para
una versión de producción habría que agregar más lugares, capturas en otros
horarios y una evaluación externa por ubicación.

## Contextual bandit con DQN en C

`dqn_study_bandit.c` muestra una decisión de un paso: el estado combina 16
características extraídas por la rama visual congelada del CNN existente con
11 metadatos. La acción es `no_apto` o `apto`; la recompensa es +1 si coincide
con la etiqueta y -1 si no. Se excluyen registros `incierto`.

El CNN ya fue entrenado de forma supervisada con las etiquetas del split
`train`/`val`; el DQN también aprende de esas etiquetas como recompensa. Por eso
es una demostración de la API de contextual bandit, no una comparación justa
contra el clasificador supervisado ni evidencia de que RL sea mejor para estos
datos. El encoder visual se mantiene congelado y el split `test` del manifest
no se usa para entrenar.

El programa procesa fotos y entrena en C. Requiere libjpeg para compilar el
target opcional y los archivos del CNN existente (`manifest.csv` y checkpoint).
Primero extrae el ZIP y ejecuta (el target requiere libjpeg disponible para CMake):

```bash
unzip -q /ruta/Datos-Supabase-2026-11-18.zip -d /tmp/study-data
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensorc_dqn_study_bandit
./build/tensorc_dqn_study_bandit \
  /tmp/study-data/Datos-Supabase-2026-11-18 \
  experiments/study_classifier/results/manifest.csv \
  experiments/study_classifier/results/study_classifier_multimodal.tck
```

La primera ruta debe apuntar a la carpeta extraída que contiene `datos.csv` y
`fotos/`. En el equipo donde se midió, extrajo características de 258 fotos en
1.480 s y entrenó 8,000 episodios en 0.172 s. En las 47 filas de prueba obtuvo
89.36 % de exactitud, 93.75 % de precisión y 78.95 % de recall para `apto`, con
matriz de confusión `[[27, 1], [4, 15]]`.

Para entrenar el DQN y clasificar otra foto en la misma ejecución, agrega estos
argumentos después del checkpoint: `PHOTO TEMP_C HUMIDITY_PCT PEOPLE HAS_TABLE
OCCUPANCY SPACE_TYPE SOUND_DBFS`. Por ejemplo:

```bash
./build/tensorc_dqn_study_bandit \
  /tmp/study-data/Datos-Supabase-2026-11-18 \
  experiments/study_classifier/results/manifest.csv \
  experiments/study_classifier/results/study_classifier_multimodal.tck \
  /ruta/foto.jpg 32 '' 0 True despejado interior -22
```

`-22` se expresa en dBFS, como el campo `sound_dbfs` del conjunto.

## Versión Python/NumPy

`dqn_study_bandit_python.py` es una implementación aparte: extrae HOG, histogramas
RGB y los mismos 11 metadatos en Python, entrena `tensorc.DQNAgent` y guarda
`policy.npz`, métricas y predicciones. No usa el CNN ni el DQN C. Con NumPy y
Pillow instalados, ejecútalo desde la raíz del repositorio:

```bash
PYTHONPATH=python python experiments/study_classifier/dqn_study_bandit_python.py \
  --zip /ruta/Datos-Supabase-2026-11-18.zip \
  --photo /ruta/IMG00026.jpg --temperature-c 32 --people 0 \
  --has-table true --occupancy despejado --space-type interior --sound-dbfs -22
```

Es un contextual bandit sintético de un paso que usa etiquetas conocidas como
recompensa; sirve de ejemplo de API y no equivale a aprender una política de
interacción real. `select_action` produce una clase, no probabilidad calibrada.
