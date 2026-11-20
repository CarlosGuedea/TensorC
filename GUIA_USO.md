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

layer = tensorc.DenseLayer(2, 2, seed=7)
clf = tensorc.TensorCClassifier(
    forward=lambda value: layer.forward(value),
    params=[layer.weight, layer.bias],
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

La función forward debe recibir un tensor TensorC y devolver logits con forma
[muestras, clases]. El clasificador aplica softmax y cross-entropy.

## 9. MNIST completo

TensorFlow solo se utiliza para descargar MNIST. La red, el autodiferenciador
y Adam son de TensorC.

~~~bash
python -m pip install tensorflow
~~~

~~~python
import numpy as np
import tensorflow as tf
import tensorc

(X_train, y_train), (X_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

X_train = np.ascontiguousarray(
    X_train[:, None, :, :].astype(np.float32) / 255.0
)
X_test = np.ascontiguousarray(
    X_test[:, None, :, :].astype(np.float32) / 255.0
)

class MNISTNetwork:
    def __init__(self):
        self.conv1 = tensorc.Conv2DLayer(
            1, 8, 3, 3, padding=(1, 1), seed=101
        )
        self.conv2 = tensorc.Conv2DLayer(
            8, 16, 3, 3, padding=(1, 1), seed=202
        )
        self.output = tensorc.DenseLayer(16 * 7 * 7, 10, seed=303)
        self.params = [
            self.conv1.kernel, self.conv1.bias,
            self.conv2.kernel, self.conv2.bias,
            self.output.weight, self.output.bias,
        ]

    def forward(self, x):
        x = self.conv1.forward(x).relu().maxpool2d(2, 2)
        x = self.conv2.forward(x).relu().maxpool2d(2, 2)
        x = x.reshape((int(x.shape[0]), 16 * 7 * 7))
        return self.output.forward(x)

network = MNISTNetwork()
clf = tensorc.TensorCClassifier(
    forward=network.forward, params=network.params,
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

## 15. Enlaces

- [Repositorio TensorC](https://github.com/CarlosGuedea/TensorC)
- [TensorC en PyPI](https://pypi.org/project/tensorc/)
- [Documentación Python](python/README.md)
- [Ejemplos](examples/)
- [Issues](https://github.com/CarlosGuedea/TensorC/issues)

