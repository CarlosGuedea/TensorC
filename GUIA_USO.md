# Manual de uso de TensorC

TensorC es un framework de redes neuronales con núcleo C11 y bindings para
Python. Este manual cubre instalación, tensores, entrenamiento, clasificación,
MNIST, aprendizaje por refuerzo y la API C. TensorFlow y PyTorch no son
requisitos.

## Contenido

1. [Qué es TensorC](#1-qué-es-tensorc)
2. [Instalación](#2-instalación)
3. [Verificar la instalación](#3-verificar-la-instalación)
4. [Tensores y autodiferenciación](#4-tensores-y-autodiferenciación)
5. [Regresión lineal con Adam](#5-regresión-lineal-con-adam)
6. [Entrenamiento con Model](#6-entrenamiento-con-model)
7. [Capas densas y convolucionales](#7-capas-densas-y-convolucionales)
8. [Wrapper estilo scikit-learn](#8-wrapper-estilo-scikit-learn)
9. [MNIST completo](#9-mnist-completo)
10. [Clasificador de lugares para estudiar](#10-clasificador-de-lugares-para-estudiar)
11. [Guardar y cargar pesos](#11-guardar-y-cargar-pesos)
12. [Compilar y usar TensorC desde C11](#12-compilar-y-usar-tensorc-desde-c11)
13. [Problemas frecuentes](#13-problemas-frecuentes)
14. [Medición de rendimiento](#14-medición-de-rendimiento)
15. [Aprendizaje por refuerzo](#15-aprendizaje-por-refuerzo)
16. [Enlaces](#16-enlaces)

## 1. ¿Qué es TensorC?

TensorC es un framework pequeño de redes neuronales con núcleo numérico en C11.
Con `TensorCClassifier(model=Sequential(...))`, las operaciones, el autograd,
Adam y el entrenamiento por lotes se ejecutan en C. Python controla el programa
y puede cargar y preparar datos. La API experimental `tensorc.DQNAgent` es una
implementación independiente en Python/NumPy.

Incluye:

- tensores N-dimensionales y autodiferenciación;
- broadcasting, matmul, reshape, ReLU, sigmoid, tanh y softmax;
- convolución 2D, max pooling y average pooling;
- capas DenseLayer, Conv2DLayer, LSTMLayer y TransformerBlock;
- optimizadores SGD y Adam;
- Model y TensorCClassifier;
- guardado y carga de checkpoints;
- API nativa C11 y bindings para Python.

## 2. Instalación

Se recomienda utilizar un entorno virtual.

### macOS y Linux

~~~bash
mkdir prueba_tensorc
cd prueba_tensorc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorc
~~~

### Windows PowerShell

~~~powershell
mkdir prueba_tensorc
cd prueba_tensorc
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorc
~~~

Usa python -m pip en lugar de pip para garantizar que el paquete se instala
en el mismo intérprete que ejecutará el programa.

NumPy se instala como dependencia de TensorC. Los ejemplos con imágenes
requieren Pillow; el notebook clásico agrega scikit-learn y joblib:

~~~bash
python -m pip install pillow
python -m pip install scikit-learn joblib
~~~

### Código más reciente de GitHub

PyPI instala la versión publicada. Para usar funciones añadidas después de la
publicación, como `tensorc.DQNAgent`, instala `main` directamente:

~~~bash
python -m pip install "git+https://github.com/CarlosGuedea/TensorC.git@main#subdirectory=python"
~~~

Para desarrollar desde un clon, ejecuta `python -m pip install -e ./python`
desde la raíz del repositorio.

### Backend Metal opcional (macOS Apple Silicon)

La rueda normal de PyPI usa CPU. Para compilar la extensión local con Metal,
desde la raíz del clon y con el entorno virtual activado:

~~~bash
TENSORC_METAL=1 python -m pip install --no-build-isolation --no-deps \
  --force-reinstall ./python
python -c "import tensorc; print('Metal:', tensorc.metal_available())"
~~~

Requiere macOS y Xcode. Metal acelera algunas operaciones de CNN; el resto
usa CPU. El backend está en etapa inicial y todavía no garantiza acelerar el
entrenamiento completo: en una prueba MNIST de 8,192 imágenes por 4 épocas,
Metal y CPU tardaron cerca de 47 s. No hace falta TensorFlow ni PyTorch.

## 3. Verificar la instalación

~~~bash
python -c "import tensorc; print(tensorc.__version__)"
~~~

La versión esperada es:

~~~text
0.17.0
~~~

Prueba también una operación con autodiferenciación:

~~~bash
python - <<'PY'
import tensorc

x = tensorc.Tensor([1, 2, 3, 4], shape=(2, 2), requires_grad=True)
w = tensorc.Tensor([1, 0, 0, 1], shape=(2, 2), requires_grad=True)
loss = x.matmul(w).relu().sum()
loss.backward()

print("TensorC:", tensorc.__version__)
print("Pérdida:", loss.data[0])
print("Gradiente:", x.grad)
PY
~~~

## 4. Tensores y autodiferenciación

~~~python
import tensorc

x = tensorc.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = tensorc.Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

loss = x.matmul(w).relu().sum()
loss.backward()

print(loss.data)
print(x.grad)
~~~

El flujo general es:

~~~text
crear tensores -> forward -> pérdida -> backward() -> optimizer.step()
~~~

## 5. Regresión lineal con Adam

~~~python
import tensorc

x = tensorc.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = tensorc.Tensor([[2.0], [4.0], [6.0], [8.0]])

w = tensorc.Tensor([[0.1]], requires_grad=True)
b = tensorc.Tensor([[0.0]], requires_grad=True)
optimizer = tensorc.Adam([w, b], 0.05)

for _ in range(200):
    prediction = x.matmul(w).add(b)
    loss = prediction.mse_loss(y)
    loss.backward()
    optimizer.step()

print("w:", w.data)
print("b:", b.data)
print("loss:", loss.data[0])
~~~

## 6. Entrenamiento con Model

Model recibe un forward, sus parámetros y una pérdida.

~~~python
import tensorc

x = tensorc.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = tensorc.Tensor([[2.0], [4.0], [6.0], [8.0]])
w = tensorc.Tensor([[0.1]], requires_grad=True)
b = tensorc.Tensor([[0.0]], requires_grad=True)

model = tensorc.Model(
    forward=lambda value: value.matmul(w).add(b),
    params=[w, b],
    loss="mse",
    learning_rate=0.05,
)

history = model.fit(
    x, y, epochs=200, batch_size=2,
    shuffle=True, verbose=False,
)

print("predicción:", model.predict(x).data)
print("error:", model.evaluate(x, y))
model.save_weights("regresion.tck")
model.load_weights("regresion.tck")
~~~

Las pérdidas disponibles son mse, cross_entropy o una función personalizada
con firma loss(prediction, target).

## 7. Capas densas y convolucionales

### Capa densa

~~~python
import numpy as np
import tensorc

layer = tensorc.DenseLayer(4, 3, seed=7)
x = tensorc.Tensor(np.random.rand(8, 4).astype(np.float32))
y = layer.forward(x)
print(y.shape)  # (8, 3)
~~~

### Conv2D

Las imágenes utilizan formato NCHW:

~~~text
[lote, canales, alto, ancho]
~~~

~~~python
import numpy as np
import tensorc

conv = tensorc.Conv2DLayer(
    in_channels=3, out_channels=16,
    kernel_h=3, kernel_w=3,
    stride=(1, 1), padding=(1, 1), seed=42,
)

images = np.random.rand(8, 3, 32, 32).astype(np.float32)
features = conv.forward(tensorc.Tensor(images))
print(features.shape)  # (8, 16, 32, 32)
~~~

Los parámetros entrenables son layer.weight, layer.bias, conv.kernel y
conv.bias.

Para ejemplos de `LSTMLayer` (entrada `[seq, batch, in]`) y
`TransformerBlock` (entrada `[batch, seq, model_dim]`), consulta los
[ejemplos Python](python/README.md#ejemplos).

## 8. Wrapper estilo scikit-learn

TensorCClassifier ofrece fit, predict, predict_proba, score, get_params,
set_params, save y load_weights.

~~~python
import numpy as np
import tensorc

X = np.array([
    [0.0, 0.0], [0.0, 1.0],
    [1.0, 0.0], [1.0, 1.0],
], dtype=np.float32)
y = np.array([0, 1, 1, 1])

network = tensorc.Sequential([
    tensorc.DenseLayer(2, 8, seed=7),
    tensorc.ReLULayer(),
    tensorc.DenseLayer(8, 2, seed=8),
])
clf = tensorc.TensorCClassifier(
    model=network,
    epochs=300,
    batch_size=4,
    learning_rate=0.05,
    random_state=42, verbose=False,
)

clf.fit(X, y)
print("predicciones:", clf.predict(X))
print("probabilidades:", clf.predict_proba(X))
print("exactitud:", clf.score(X, y))
clf.save("clasificador.tck")

# Recarga en otro proceso: reconstruye la misma arquitectura.
network2 = tensorc.Sequential([
    tensorc.DenseLayer(2, 8, seed=7),
    tensorc.ReLULayer(),
    tensorc.DenseLayer(8, 2, seed=8),
])
clf2 = tensorc.TensorCClassifier(model=network2, verbose=False)
clf2.load_weights("clasificador.tck")
print("exactitud recargada:", clf2.score(X, y))
~~~

Con `model=Sequential(...)`, el forward, la pérdida, el entrenamiento por
lotes y la predicción se ejecutan en C. La forma anterior con `forward` y
`params` sigue disponible para modelos personalizados y conserva su bucle de
entrenamiento en Python:

~~~python
import numpy as np
import tensorc

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_train = np.array([0, 1, 1, 1])
layer = tensorc.DenseLayer(2, 2, seed=7)
clf = tensorc.TensorCClassifier(
    forward=lambda x: layer.forward(x),
    params=[layer.weight, layer.bias],
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
)
clf.fit(X_train, y_train)
~~~

## 9. MNIST completo

El ejemplo lee MNIST desde `~/.keras/datasets/mnist.npz`; no requiere Keras ni
TensorFlow. Descarga el archivo de datos con:

~~~bash
mkdir -p ~/.keras/datasets
curl -L https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz \
  -o ~/.keras/datasets/mnist.npz
~~~

El archivo contiene 60,000 imágenes de entrenamiento y 10,000 de prueba; la
[documentación de Keras MNIST](https://keras.io/api/datasets/mnist/) describe
su formato. NumPy solo abre el archivo. `preprocess_mnist_images` convierte
las imágenes uint8 a float32, normaliza sus valores y agrega el canal en C.
El entrenamiento y la inferencia de `Sequential` también se ejecutan en C.

Para comparar solo la preparación Python/NumPy frente a C:

~~~bash
python examples/benchmark_mnist_preprocessing.py
~~~

El benchmark usa las 60,000 imágenes de entrenamiento y muestra la mediana de
cinco ejecuciones por ruta; no incluye carga del archivo ni entrenamiento.

Para comparar entrenamiento e inferencia de la misma CNN con NumPy y TensorC:

~~~bash
python examples/benchmark_mnist_cnn_numpy.py
~~~

Este ejemplo usa 8,192 imágenes para entrenar, 2,048 para probar y comparte
pesos iniciales, orden de minibatches y número de épocas entre ambas rutas.

~~~python
import numpy as np
import tensorc
from pathlib import Path

with np.load(Path.home() / ".keras/datasets/mnist.npz") as mnist:
    X_train, y_train = mnist["x_train"], mnist["y_train"]
    X_test, y_test = mnist["x_test"], mnist["y_test"]

X_train = tensorc.preprocess_mnist_images(X_train)
X_test = tensorc.preprocess_mnist_images(X_test)

network = tensorc.Sequential([
    tensorc.Conv2DLayer(1, 8, 3, 3, padding=(1, 1), seed=101),
    tensorc.ReLULayer(),
    tensorc.MaxPool2DLayer(2, 2),
    tensorc.Conv2DLayer(8, 16, 3, 3, padding=(1, 1), seed=202),
    tensorc.ReLULayer(),
    tensorc.MaxPool2DLayer(2, 2),
    tensorc.FlattenLayer(),
    tensorc.DenseLayer(16 * 7 * 7, 10, seed=303),
])
clf = tensorc.TensorCClassifier(
    model=network,
    epochs=5, batch_size=64,
    learning_rate=0.001, random_state=42,
)

clf.fit(X_train, y_train)
print("Exactitud:", clf.score(X_test, y_test))
clf.save("mnist_tensorc.tck")
~~~

En una prueba de referencia, 60,000 imágenes y 5 épocas tardaron 93.63 segundos
en un Mac ARM. El tiempo varía según el equipo y los parámetros.

## 10. Clasificador de lugares para estudiar

Este experimento tiene dos entradas: fotografía y 11 metadatos. El wrapper
genérico recibe una sola matriz, por lo que se utiliza StudyNetwork:

~~~bash
python experiments/study_classifier/train_study_classifier.py \
  --zip Datos-Supabase-2026-11-18.zip \
  --output experiments/study_classifier/results/python \
  --epochs 35 \
  --batch-size 16
~~~

El script divide las filas etiquetadas en entrenamiento, validación y prueba;
las filas `incierto` se predicen aparte. En el directorio de salida guarda el
mejor checkpoint, el historial de épocas, las métricas de prueba y ambos
archivos de predicciones.

La arquitectura es:

~~~text
imagen: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Dense
metadatos: Dense
fusión: concatenación -> Dense(2)
~~~

Este entrenamiento multimodal usa el núcleo TensorC en C. Para el ejemplo
supervisado clásico en Python/scikit-learn, abre
[train_study_classifier_classic.ipynb](experiments/study_classifier/train_study_classifier_classic.ipynb);
extrae características HOG y color, añade metadatos y guarda un modelo `.joblib`.

## 11. Guardar y cargar pesos

~~~python
import tensorc

tensorc.save([w, b], "checkpoint.tck")
weights = tensorc.load("checkpoint.tck", requires_grad=True)

x.save("tensor.tcb")
restored = tensorc.load("tensor.tcb", requires_grad=True)
~~~

Para cargar un modelo se debe reconstruir la misma arquitectura. TensorC
valida el número de tensores y sus formas antes de copiarlos. `Model` usa
`save_weights` y `load_weights`; `TensorCClassifier.save` guarda además las
clases y metadatos del estimador en un archivo `.json` contiguo.

## 12. Compilar y usar TensorC desde C11

Requisitos: CMake 3.16 o superior, compilador C11 y Git.

~~~bash
git clone https://github.com/CarlosGuedea/TensorC.git
cd TensorC
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
~~~

Ejemplos:

~~~bash
./build/tensorc_xor
./build/tensorc_fahrenheit
./build/tensorc_autograd
./build/tensorc_cnn_mnist
~~~

Programa mínimo:

~~~c
#include <tensorc/tensorc.h>
#include <stdio.h>

int main(void) {
    const size_t shape[] = {2, 2};
    const float values[] = {1.0f, -2.0f, 3.0f, 4.0f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *y = tc_tensor_nd_relu(x);
    tc_tensor_nd *loss = tc_tensor_nd_sum(y);

    tc_tensor_nd_backward(loss);
    printf("loss = %f\n", tc_tensor_nd_const_data(loss)[0]);

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(y);
    tc_tensor_nd_free(x);
    return 0;
}
~~~

## 13. Problemas frecuentes

### TensorC no se encuentra

~~~bash
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade tensorc
~~~

En VS Code selecciona el mismo intérprete:

~~~text
Cmd + Shift + P -> Python: Select Interpreter -> .venv/bin/python
~~~

### No aparece salida

~~~bash
python -u prueba.py
~~~

Agrega mensajes antes de pasos lentos:

~~~python
print("Cargando dataset...", flush=True)
~~~

Guarda el archivo con Cmd + S antes de ejecutarlo.

### Forma incorrecta en Conv2D

~~~python
# Escala de grises
X = X.astype("float32") / 255.0
X = X[:, None, :, :]

# RGB en formato NHWC
X = X.transpose(0, 3, 1, 2)
~~~

### Falta de memoria

~~~python
X_train = X_train[:10000]
y_train = y_train[:10000]
batch_size = 32
~~~

## 14. Medición de rendimiento

~~~python
import time

start = time.perf_counter()
clf.fit(X_train, y_train)
elapsed = time.perf_counter() - start
print(f"Entrenamiento: {elapsed:.3f} segundos")
~~~

Separa carga, preparación, entrenamiento, evaluación y guardado. La descarga
inicial de MNIST no representa el tiempo de entrenamiento.

## 15. Aprendizaje por refuerzo

### API DQN en C11

TensorC incluye una API C experimental de DQN para observaciones vectoriales y
acciones discretas. El agente ofrece una MLP densa, replay uniforme, red
objetivo, epsilon-greedy, TD(0), MSE y Adam. El programa C del usuario conserva
el ciclo del entorno y llama a `tc_dqn_select_action`, `tc_dqn_remember` y
`tc_dqn_train_step`; este último devuelve `TC_DQN_TRAIN_NOT_READY` hasta llenar
el primer minibatch y sincroniza la red objetivo automáticamente según la
configuración. Para el entorno, `tc_env` ofrece una interfaz C estilo Gymnasium:
observaciones como vector `float`, acciones discretas, `tc_env_reset` y
`tc_env_step`.

~~~c
tc_env env = {
    .observation_size = 4,
    .action_count = 2,
    .user_data = &mi_estado,
    .reset = mi_reset,
    .step = mi_step,
    .close = mi_close, /* opcional */
};
float observation[4], reward;
int terminated, truncated;
tc_env_reset(&env, seed, observation);
tc_env_step(&env, action, observation, &reward, &terminated, &truncated);
~~~

Ambas señales terminan el episodio. Pasa `terminated` como el último argumento
de `tc_dqn_remember`: una terminación real detiene el bootstrap y un límite de
tiempo (`truncated`) conserva el bootstrap. El callback `close` es opcional y
`tc_env_close` invalida el descriptor después de cerrarlo. El descriptor y su
estado son propiedad del llamador. Esta API no carga entornos de Gymnasium para
Python; permite implementar entornos con un contrato similar directamente en C.

Compila y ejecuta el ejemplo CartPole, cuya dinámica, agente y entrenamiento
son C11:

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensorc_dqn_cartpole
./build/tensorc_dqn_cartpole
~~~

La API C no cubre acciones continuas ni PPO/actor-critic. El ejemplo implementa
CartPole dentro del mismo archivo C11, sin instalar Gymnasium, y reporta
recompensa media y tiempo de entrenamiento.

### Ejemplo con lugares y fotos

`experiments/study_classifier/dqn_study_bandit.c` usa el CNN multimodal
existente como extractor visual congelado. Para cada lugar combina su embedding
visual de 16 valores con 11 metadatos y el DQN elige `no_apto` o `apto`. Una
decisión correcta recibe +1; una incorrecta, -1. Es un contextual bandit de un
paso porque el ZIP contiene etiquetas, no trayectorias. El CNN se entrenó con
esas mismas etiquetas, así que el ejemplo ilustra la API RL y no demuestra que
RL supere al entrenamiento supervisado.

El target requiere libjpeg y está disponible si CMake encuentra esa biblioteca.
Extrae el ZIP y ejecuta:

~~~bash
unzip -q /ruta/Datos-Supabase-2026-11-18.zip -d /tmp/study-data
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensorc_dqn_study_bandit
./build/tensorc_dqn_study_bandit \
  /tmp/study-data/Datos-Supabase-2026-11-18 \
  experiments/study_classifier/results/manifest.csv \
  experiments/study_classifier/results/study_classifier_multimodal.tck
~~~

La carpeta del dataset debe contener `datos.csv` y `fotos/`. El programa usa el
split `train` para entrenamiento y `test` para evaluación; ignora `incierto`.
En una corrida sobre las 47 filas de prueba obtuvo 89.36 % de exactitud, 93.75 %
de precisión, 78.95 % de recall y F1 de 85.71 % para `apto`; la matriz fue
`[[27,1],[4,15]]`. Consulta el [README del experimento](experiments/study_classifier/README.md)
para metodología, tiempos y limitaciones.

Para clasificar una foto nueva durante la misma ejecución DQN, agrega después
del checkpoint estos argumentos: `PHOTO TEMP_C HUMIDITY_PCT PEOPLE HAS_TABLE
OCCUPANCY SPACE_TYPE SOUND_DBFS`. Ejemplo para una foto interior con mesa,
despejada, 32 °C, cero personas y −22 dBFS:

~~~bash
./build/tensorc_dqn_study_bandit \
  /tmp/study-data/Datos-Supabase-2026-11-18 \
  experiments/study_classifier/results/manifest.csv \
  experiments/study_classifier/results/study_classifier_multimodal.tck \
  /ruta/foto.jpg 32 '' 0 True despejado interior -22
~~~

El DQN devuelve la acción `apto`/`no_apto`, no una probabilidad calibrada.

### Agente DQN en Python/NumPy

`tensorc.DQNAgent` acepta observaciones vectoriales y acciones discretas. El
agente y sus cálculos corren en Python/NumPy; no requiere TensorFlow ni
Gymnasium. Para conectarlo a un entorno Gymnasium existente, instala esa
dependencia opcional:

~~~bash
python -m pip install "gymnasium[classic-control]"
~~~

Ejemplo con CartPole:

~~~python
import gymnasium as gym
import tensorc

env = gym.make("CartPole-v1")
agent = tensorc.DQNAgent(
    observation_size=4,
    action_count=2,
    hidden_size=64,
    replay_capacity=10_000,
    batch_size=64,
    target_update_interval=250,
    learning_rate=0.001,
    discount_factor=0.99,
    seed=42,
)

for episode in range(200):
    state, _ = env.reset(seed=42 + episode)
    epsilon = max(0.05, 1.0 - episode / 200)
    finished = False
    while not finished:
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.remember(
            state, action, reward, next_state,
            terminated=terminated, truncated=truncated,
        )
        loss = agent.train_step()  # None durante el warm-up del replay
        state = next_state
        finished = terminated or truncated

env.close()
agent.save("cartpole_policy.npz")
agent_cargado = tensorc.DQNAgent.load("cartpole_policy.npz")
~~~

`terminated=True` detiene el bootstrap; `truncated=True` cierra el episodio,
pero conserva el bootstrap. `save` guarda pesos de inferencia, no el replay ni
el estado de Adam; cargarlo no reanuda la sesión de entrenamiento. La API
Python no incluye acciones continuas ni una clase de entorno.

### Bandit contextual con el ZIP de lugares

`experiments/study_classifier/dqn_study_bandit_python.py` lee el ZIP y extrae
HOG, color y 11 metadatos en Python; no usa el CNN ni el DQN C:

~~~bash
PYTHONPATH=python python experiments/study_classifier/dqn_study_bandit_python.py \
  --zip /ruta/Datos-Supabase-2026-11-18.zip \
  --photo /ruta/IMG00026.jpg --temperature-c 32 --people 0 \
  --has-table true --occupancy despejado --space-type interior --sound-dbfs -22
~~~

Guarda `policy.npz`, métricas y predicciones en `results/python_dqn/`. Al ser
una decisión de un paso entrenada con las etiquetas del ZIP, es un ejemplo de
contextual bandit, no evidencia de aprendizaje de una política interactiva.

## 16. Enlaces

- [Repositorio TensorC](https://github.com/CarlosGuedea/TensorC)
- [TensorC en PyPI](https://pypi.org/project/tensorc/)
- [Documentación Python](python/README.md)
- [Ejemplos](examples/)
- [Issues](https://github.com/CarlosGuedea/TensorC/issues)

