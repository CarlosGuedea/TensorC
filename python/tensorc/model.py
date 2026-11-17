"""High-level training API: Model with fit/evaluate/predict/weights.

Pure Python over the C extension: a Model is just a forward callable plus
the list of trainable parameter tensors; the autograd lives in the C core.
"""

import random

from . import Adam, Tensor, load, save

_LOSSES = {
    "mse": lambda prediction, target: prediction.mse_loss(target),
    "cross_entropy": lambda prediction, target: prediction.cross_entropy(target),
}


def _gather_rows(tensor, rows):
    """Reúne filas arbitrarias de un tensor [N, ...] vía slice + concat.

    Es el "gather" del v1: una slice por fila y un concat las une. Con
    batch sizes razonables el costo es aceptable y mantiene el autograd
    intacto (slice y concat tienen gradiente).
    """
    parts = [tensor.slice_axis(0, int(r), 1) for r in rows]
    if len(parts) == 1:
        return parts[0]
    return parts[0].concat(parts[1:], axis=0)


class Model:
    """Envuelve un forward + sus parámetros entrenables.

    Args:
        forward: callable(Tensor) -> Tensor con las predicciones.
        params: lista de ``tensorc.Tensor`` entrenables del modelo. Los
            gradientes llegan a ellos vía ``backward()`` desde la loss.
        loss: ``'mse'``, ``'cross_entropy'`` o callable(pred, target) -> Tensor.
        optimizer: objeto con ``.step()`` (p. ej. ``tensorc.Adam``). Por
            defecto ``Adam(params, learning_rate)``.
        learning_rate: tasa del Adam por defecto.
        seed: semilla del shuffle de mini-batches.

    Example:
        >>> w = tc.Tensor([0.0], shape=(1, 1), requires_grad=True)
        >>> b = tc.Tensor([0.0], shape=(1, 1), requires_grad=True)
        >>> model = tc.Model(lambda x: x.matmul(w).add(b), [w, b], loss="mse",
        ...                  learning_rate=0.1)
        >>> history = model.fit(x, y, epochs=200, verbose=False)
    """

    def __init__(self, forward, params, loss="mse", optimizer=None,
                 learning_rate=0.01, seed=1):
        if not callable(forward):
            raise TypeError("forward debe ser callable")
        self.forward = forward
        self.params = list(params)
        if isinstance(loss, str):
            if loss not in _LOSSES:
                raise ValueError(
                    "loss desconocida: %r (usa 'mse', 'cross_entropy' o callable)"
                    % (loss,))
            self._loss = _LOSSES[loss]
        elif callable(loss):
            self._loss = loss
        else:
            raise TypeError("loss debe ser str o callable")
        self.optimizer = (optimizer if optimizer is not None
                          else Adam(self.params, learning_rate))
        self._rng = random.Random(seed)

    def _batches(self, x, y, batch_size, shuffle):
        """Genera (xb, yb, filas) cubriendo el eje 0 de x e y."""
        n = int(x.shape[0])
        if n == 0:
            raise ValueError("x tiene 0 filas")
        if int(y.shape[0]) != n:
            raise ValueError("x e y deben tener el mismo número de filas")
        bs = int(batch_size) if batch_size else n
        if bs <= 0:
            raise ValueError("batch_size debe ser positivo")
        order = list(range(n))
        if shuffle and bs < n:
            self._rng.shuffle(order)
        for start in range(0, n, bs):
            rows = order[start:start + bs]
            if shuffle and bs < n:
                yield _gather_rows(x, rows), _gather_rows(y, rows), len(rows)
            else:
                length = len(rows)
                yield (x.slice_axis(0, start, length),
                       y.slice_axis(0, start, length), length)

    def fit(self, x, y, epochs=100, batch_size=None, shuffle=True,
            verbose=True, log_every=10):
        """Entrena el modelo. Devuelve history con la pérdida por época."""
        history = {"loss": []}
        for epoch in range(int(epochs)):
            total, seen = 0.0, 0
            for xb, yb, rows in self._batches(x, y, batch_size, shuffle):
                prediction = self.forward(xb)
                loss = self._loss(prediction, yb)
                loss.backward()
                self.optimizer.step()
                total += float(loss.data[0]) * rows
                seen += rows
            average = total / seen
            history["loss"].append(average)
            if verbose and (epoch % int(log_every) == 0 or epoch == epochs - 1):
                print("epoch %d/%d - loss: %.6f" % (epoch + 1, epochs, average))
        return history

    def evaluate(self, x, y, batch_size=None):
        """Pérdida promedio sobre (x, y) sin entrenar."""
        total, seen = 0.0, 0
        for xb, yb, rows in self._batches(x, y, batch_size, shuffle=False):
            loss = self._loss(self.forward(xb), yb)
            total += float(loss.data[0]) * rows
            seen += rows
        return total / seen

    def predict(self, x):
        """Ejecuta el forward sin entrenar."""
        return self.forward(x)

    def save_weights(self, path):
        """Guarda los parámetros como checkpoint binario (ver tensorc.save)."""
        save(self.params, path)

    def load_weights(self, path):
        """Carga los parámetros desde un checkpoint guardado con save_weights.

        Valida el número de tensores y las formas; copia los datos en los
        parámetros existentes (copy_), así el optimizador sigue válido.
        """
        tensors = load(path, requires_grad=True)
        if isinstance(tensors, Tensor):
            tensors = [tensors]
        if len(tensors) != len(self.params):
            raise ValueError(
                "el checkpoint tiene %d tensores y el modelo %d"
                % (len(tensors), len(self.params)))
        for param, saved in zip(self.params, tensors):
            if list(param.shape) != list(saved.shape):
                raise ValueError(
                    "forma distinta en load_weights: %s vs %s"
                    % (list(param.shape), list(saved.shape)))
            param.copy_(saved)
