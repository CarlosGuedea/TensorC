"""Interfaz de clasificación estilo scikit-learn sobre el núcleo TensorC.

La API acepta arrays NumPy, pero las operaciones de red, autograd y Adam
siguen ejecutándose en el núcleo C expuesto por la extensión TensorC.
"""

import json
import random
from pathlib import Path

import numpy as np

from . import Adam, Tensor, load, save


def _as_float32_array(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[0] == 0:
        raise ValueError("%s debe tener al menos una muestra" % name)
    return np.ascontiguousarray(array, dtype=np.float32)


def _one_hot(labels, classes):
    result = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        result[row, int(label)] = 1.0
    return result


class TensorCClassifier:
    """Clasificador TensorC con una interfaz parecida a scikit-learn.

    `forward` debe recibir un `tensorc.Tensor` y devolver logits [batch, clases].
    `params` es la lista de tensores entrenables del modelo.

    Example:
        >>> layer = tensorc.DenseLayer(4, 2, seed=7)
        >>> clf = TensorCClassifier(
        ...     forward=lambda x: layer.forward(x),
        ...     params=[layer.weight, layer.bias],
        ...     epochs=100,
        ... )
        >>> clf.fit(X_train, y_train)
        >>> predicciones = clf.predict(X_test)
    """

    def __init__(self, forward, params, *, epochs=100, batch_size=32,
                 learning_rate=0.001, shuffle=True, random_state=1,
                 verbose=True, log_every=10):
        if not callable(forward):
            raise TypeError("forward debe ser callable")
        if not params:
            raise ValueError("params no puede estar vacío")
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("epochs, batch_size y learning_rate deben ser positivos")
        self.forward = forward
        self.params = list(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.log_every = int(log_every)
        self.optimizer = Adam(self.params, self.learning_rate)
        self.history_ = {"loss": []}
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.is_fitted_ = False

    def get_params(self, deep=True):
        """Devuelve hiperparámetros con la convención de scikit-learn."""
        return {
            "forward": self.forward,
            "params": self.params,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "log_every": self.log_every,
        }

    def set_params(self, **params):
        """Actualiza hiperparámetros y devuelve el estimador."""
        allowed = set(self.get_params())
        unknown = set(params) - allowed
        if unknown:
            raise ValueError("Parámetros desconocidos: %s" % sorted(unknown))
        for name, value in params.items():
            setattr(self, name, value)
        if "learning_rate" in params:
            self.optimizer.set_lr(float(self.learning_rate))
        return self

    def _check_y(self, y, fit=False):
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y debe ser un vector de etiquetas 1D")
        if fit:
            classes = np.unique(labels)
            if len(classes) < 2:
                raise ValueError("y debe contener al menos dos clases")
            self.classes_ = classes
        if self.classes_ is None:
            raise RuntimeError("el clasificador todavía no está entrenado")
        mapping = {value: index for index, value in enumerate(self.classes_)}
        try:
            encoded = np.asarray([mapping[value] for value in labels], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("y contiene una clase no conocida") from exc
        return labels, encoded

    def _check_X(self, X, fit=False):
        array = _as_float32_array(X, "X")
        if fit:
            self.n_features_in_ = array.shape[1] if array.ndim > 1 else 1
        return array

    def _predict_tensor(self, X):
        tensor = Tensor(X)
        return self.forward(tensor)

    def fit(self, X, y):
        """Entrena el estimador y devuelve `self`."""
        X = self._check_X(X, fit=True)
        labels, encoded = self._check_y(y, fit=True)
        if len(X) != len(labels):
            raise ValueError("X e y deben tener el mismo número de muestras")
        if len(self.classes_) != 2 and len(self.classes_) < 2:
            raise ValueError("se requieren al menos dos clases")

        target = _one_hot(encoded, self.classes_)
        rng = random.Random(self.random_state)
        self.history_ = {"loss": []}
        for epoch in range(self.epochs):
            order = list(range(len(X)))
            if self.shuffle:
                rng.shuffle(order)
            total_loss = 0.0
            seen = 0
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                xb = np.ascontiguousarray(X[rows], dtype=np.float32)
                yb = Tensor(np.ascontiguousarray(target[rows], dtype=np.float32))
                logits = self._predict_tensor(xb)
                loss = logits.softmax().cross_entropy(yb)
                loss.backward()
                self.optimizer.step()
                count = len(rows)
                total_loss += float(loss.data[0]) * count
                seen += count
            average = total_loss / seen
            self.history_["loss"].append(average)
            self.n_iter_ = epoch + 1
            if self.verbose and (epoch % self.log_every == 0 or epoch == self.epochs - 1):
                print("epoch %d/%d - loss: %.6f" %
                      (epoch + 1, self.epochs, average))
        self.is_fitted_ = True
        return self

    def _check_fitted(self):
        if not self.is_fitted_ or self.classes_ is None:
            raise RuntimeError("llama primero a fit(X, y)")

    def predict_proba(self, X):
        """Devuelve probabilidades NumPy con forma [muestras, clases]."""
        self._check_fitted()
        X = self._check_X(X)
        probabilities = []
        for start in range(0, len(X), self.batch_size):
            logits = self._predict_tensor(X[start:start + self.batch_size])
            batch = np.asarray(logits.softmax().data, dtype=np.float32)
            probabilities.append(batch.reshape(-1, len(self.classes_)))
        return np.concatenate(probabilities, axis=0)

    def predict(self, X):
        """Devuelve las etiquetas predichas en el mismo tipo que `classes_`."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y):
        """Devuelve exactitud, como los clasificadores de scikit-learn."""
        expected, _ = self._check_y(y)
        predicted = self.predict(X)
        if len(expected) != len(predicted):
            raise ValueError("X e y deben tener el mismo número de muestras")
        return float(np.mean(predicted == expected))

    def save(self, path):
        """Guarda pesos TensorC y metadatos del estimador."""
        self._check_fitted()
        path = Path(path)
        save(self.params, str(path))
        metadata = {
            "classes": self.classes_.tolist(),
            "n_features_in": self.n_features_in_,
            "n_iter": self.n_iter_,
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        return self

    def load_weights(self, path):
        """Carga pesos previamente guardados en el mismo estimador."""
        path = Path(path)
        tensors = load(str(path), requires_grad=True)
        if isinstance(tensors, Tensor):
            tensors = [tensors]
        if len(tensors) != len(self.params):
            raise ValueError("el checkpoint no coincide con params")
        for parameter, saved in zip(self.params, tensors):
            if tuple(parameter.shape) != tuple(saved.shape):
                raise ValueError("forma incompatible en el checkpoint")
            parameter.copy_(saved)
        metadata_path = path.with_suffix(path.suffix + ".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.classes_ = np.asarray(metadata["classes"])
            self.n_features_in_ = metadata.get("n_features_in")
            self.n_iter_ = metadata.get("n_iter", 0)
            self.is_fitted_ = True
        return self
