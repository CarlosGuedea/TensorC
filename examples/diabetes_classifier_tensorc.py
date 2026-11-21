"""Entrena una red TensorC para clasificar el dataset Pima Diabetes."""

import argparse
import csv
from time import perf_counter

import numpy as np
import tensorc


FEATURES = (
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
)
ZERO_IS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = set(FEATURES) | {"Outcome"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("El CSV debe incluir las 8 columnas Pima y Outcome")
    X = np.asarray([[float(row[name]) for name in FEATURES] for row in rows],
                   dtype=np.float32)
    y = np.asarray([int(row["Outcome"]) for row in rows], dtype=np.int64)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Outcome debe contener solo 0 y 1")
    return X, y


def stratified_split(y, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        count = int(np.ceil(len(indices) * test_fraction))
        test.extend(indices[:count])
        train.extend(indices[count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train), np.asarray(test)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv_path", help="ruta a diabetes.csv")
args = parser.parse_args()

X, y = load_csv(args.csv_path)
train_idx, test_idx = stratified_split(y)
X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
y_train, y_test = y[train_idx], y[test_idx]

# En estas variables, cero codifica una medición ausente; imputamos con la
# mediana del entrenamiento para evitar filtrar información del conjunto test.
zero_columns = [FEATURES.index(name) for name in ZERO_IS_MISSING]
for column in zero_columns:
    X_train[X_train[:, column] == 0, column] = np.nan
    X_test[X_test[:, column] == 0, column] = np.nan
medians = np.nanmedian(X_train, axis=0)
for column in range(X.shape[1]):
    X_train[np.isnan(X_train[:, column]), column] = medians[column]
    X_test[np.isnan(X_test[:, column]), column] = medians[column]
means = X_train.mean(axis=0)
scales = X_train.std(axis=0)
scales[scales == 0] = 1
X_train = np.ascontiguousarray((X_train - means) / scales, dtype=np.float32)
X_test = np.ascontiguousarray((X_test - means) / scales, dtype=np.float32)

network = tensorc.Sequential([
    tensorc.DenseLayer(len(FEATURES), 16, seed=7),
    tensorc.ReLULayer(),
    tensorc.DenseLayer(16, 8, seed=8),
    tensorc.ReLULayer(),
    tensorc.DenseLayer(8, 2, seed=9),
])
classifier = tensorc.TensorCClassifier(
    model=network,
    epochs=250,
    batch_size=32,
    learning_rate=0.003,
    random_state=42,
    verbose=False,
)

start = perf_counter()
classifier.fit(X_train, y_train)
training_seconds = perf_counter() - start
start = perf_counter()
probabilities = classifier.predict_proba(X_test)
evaluation_seconds = perf_counter() - start
positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
predictions = classifier.classes_[np.argmax(probabilities, axis=1)]

confusion = np.zeros((2, 2), dtype=np.int64)
for actual, predicted in zip(y_test, predictions):
    confusion[int(actual), int(predicted)] += 1
tn, fp, fn, tp = confusion.ravel()
accuracy = (tn + tp) / len(y_test)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
baseline = max(np.bincount(y_test)) / len(y_test)

print(f"TensorC {tensorc.__version__} | filas: {len(y)} | train/test: {len(train_idx)}/{len(test_idx)}")
print(f"Clases (0=no, 1=sí): {np.bincount(y).tolist()}")
print(f"Accuracy: {accuracy:.3f} | baseline mayoritaria: {baseline:.3f}")
print(f"Precision: {precision:.3f} | recall: {recall:.3f} | F1: {f1:.3f}")
print("Matriz de confusión [real, predicha]:")
print(confusion)
print(f"Entrenamiento: {training_seconds:.3f} s | evaluación: {evaluation_seconds:.3f} s")
print("Muestras (real, predicción, probabilidad de diabetes):")
for actual, predicted, row in zip(y_test[:10], predictions[:10], probabilities[:10]):
    print(f"  {int(actual)}, {int(predicted)}, {row[positive_index]:.3f}")
