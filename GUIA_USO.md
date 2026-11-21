# Guía detallada de uso de TensorC

> El proyecto se llama **TensorC**. Esta guía cubre instalación, Python,
> C11, CNN, MNIST, el wrapper estilo scikit-learn y errores frecuentes.

## 1. ¿Qué es TensorC?

TensorC es un framework pequeño de redes neuronales con un núcleo numérico
escrito en C11. Python prepara los datos y define el flujo, mientras que las
operaciones, el grafo de autodiferenciación y Adam se ejecutan en el núcleo C.

Incluye:

- tensores N-dimensionales y autodiferenciación;
- broadcasting, matmul, reshape, ReLU, sigmoid, tanh y softmax;
- convolución 2D, max pooling y average pooling;
- capas DenseLayer, Conv2DLayer, LSTMLayer y TransformerBlock;
- optimizadores SGD y Adam;
- Model y TensorCClassifier;
- guardado y carga de checkpoints;
- API nativa C11 y bindings para Python.

## 2. Instalación desde PyPI

Se recomienda utilizar un entorno virtual.

### macOS y Linux

~~~bash
mkdir prueba_tensorc
cd prueba_tensorc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorc numpy
~~~

### Windows PowerShell

~~~powershell
mkdir prueba_tensorc
cd prueba_tensorc
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorc numpy
~~~

Usa python -m pip en lugar de pip para garantizar que el paquete se instala
en el mismo intérprete que ejecutará el programa.

## 3. Verificar la instalación

~~~bash
python -c "import tensorc; print(tensorc.__version__)"
~~~

La versión esperada es:

~~~text
0.15.0
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
    epochs=300, batch_size=4,
    learning_rate=0.05,
    random_state=42, verbose=False,
)

clf.fit(X, y)
print("predicciones:", clf.predict(X))
print("probabilidades:", clf.predict_proba(X))
print("exactitud:", clf.score(X, y))
clf.save("clasificador.tck")
~~~

Con `model=Sequential(...)`, el forward, la pérdida, el entrenamiento por
lotes y la predicción se ejecutan en C. La forma anterior con `forward` y
`params` sigue disponible para modelos personalizados y conserva su bucle de
entrenamiento en Python.

## 9. MNIST completo

El ejemplo lee MNIST desde el archivo `.npz` en la caché de Keras; no importa
TensorFlow. NumPy solo abre el archivo. `preprocess_mnist_images` convierte
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

En una prueba de referencia, 60,000 imágenes y 5 épocas tardaron
aproximadamente 94 segundos en un Mac ARM.

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

La arquitectura es:

~~~text
imagen: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Dense
metadatos: Dense
fusión: concatenación -> Dense(2)
~~~

## 11. Guardar y cargar pesos

~~~python
import tensorc

tensorc.save([w, b], "checkpoint.tck")
weights = tensorc.load("checkpoint.tck", requires_grad=True)

x.save("tensor.tcb")
restored = tensorc.load("tensor.tcb", requires_grad=True)
~~~

Para cargar un modelo se debe reconstruir la misma arquitectura. TensorC
valida el número de tensores y sus formas antes de copiarlos.

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

### Error _ARRAY_API not found con TensorFlow

TensorFlow fue compilado para NumPy 1.x y el entorno tiene NumPy 2.x. Usa un
entorno limpio:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" tensorflow tensorc
~~~

Comprueba las versiones:

~~~bash
python -c "import numpy, tensorflow, tensorc; print(numpy.__version__, tensorflow.__version__, tensorc.__version__)"
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

## DQN en C11

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

La API no cubre acciones continuas ni PPO/actor-critic. El ejemplo reporta
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

### Implementación Python/NumPy

`tensorc.DQNAgent` ofrece la misma interfaz básica para observaciones vectoriales
y acciones discretas, implementada en Python/NumPy. El ejemplo
`experiments/study_classifier/dqn_study_bandit_python.py` lee el ZIP directamente
y extrae HOG, color y 11 metadatos en Python, sin usar CNN/DQN C:

~~~bash
PYTHONPATH=python python experiments/study_classifier/dqn_study_bandit_python.py \
  --zip /ruta/Datos-Supabase-2026-11-18.zip \
  --photo /ruta/IMG00026.jpg --temperature-c 32 --people 0 \
  --has-table true --occupancy despejado --space-type interior --sound-dbfs -22
~~~

Guarda `policy.npz`, métricas y predicciones en `results/python_dqn/`. Al ser
una decisión de un paso entrenada con las etiquetas del ZIP, es un ejemplo de
contextual bandit, no evidencia de aprendizaje de una política interactiva.

## 15. Enlaces

- [Repositorio TensorC](https://github.com/CarlosGuedea/TensorC)
- [TensorC en PyPI](https://pypi.org/project/tensorc/)
- [Documentación Python](python/README.md)
- [Ejemplos](examples/)
- [Issues](https://github.com/CarlosGuedea/TensorC/issues)

