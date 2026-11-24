#!/usr/bin/env python3
"""Entrena TensorC para la competencia Spaceship Titanic de Kaggle."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorc


CATEGORICAL = ("HomePlanet", "CryoSleep", "Destination", "VIP",
               "CabinDeck", "CabinSide")
NUMERIC = ("Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa",
           "VRDeck", "CabinNumber")
MISSING = "__MISSING__"
UNKNOWN = "__UNKNOWN__"


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _features(row):
    cabin = (row.get("Cabin") or "").strip().split("/")
    cabin_deck = cabin[0].strip() if len(cabin) == 3 else ""
    cabin_number = cabin[1].strip() if len(cabin) == 3 else ""
    cabin_side = cabin[2].strip() if len(cabin) == 3 else ""
    values = {
        "HomePlanet": row.get("HomePlanet"),
        "CryoSleep": row.get("CryoSleep"),
        "Destination": row.get("Destination"),
        "VIP": row.get("VIP"),
        "CabinDeck": cabin_deck,
        "CabinSide": cabin_side,
        "Age": _number(row.get("Age")),
        "RoomService": _number(row.get("RoomService")),
        "FoodCourt": _number(row.get("FoodCourt")),
        "ShoppingMall": _number(row.get("ShoppingMall")),
        "Spa": _number(row.get("Spa")),
        "VRDeck": _number(row.get("VRDeck")),
        "CabinNumber": _number(cabin_number),
    }
    for name in CATEGORICAL:
        value = values[name]
        value = str(value).strip() if value is not None else ""
        if not value or value.lower() in {"nan", "none", "null"}:
            values[name] = MISSING
        elif name in {"CryoSleep", "VIP"} and value.lower() in {"true", "false"}:
            values[name] = value.title()
        else:
            values[name] = value
    return values


class FeatureEncoder:
    """One-hot para categorías y estandarización con estadísticas de train."""

    def fit(self, rows):
        features = [_features(row) for row in rows]
        self.categories = {
            name: sorted({row[name] for row in features if row[name] != MISSING})
            + [MISSING, UNKNOWN]
            for name in CATEGORICAL
        }
        numeric = np.asarray([[row[name] for name in NUMERIC]
                              for row in features], dtype=np.float64)
        self.medians = np.zeros(len(NUMERIC), dtype=np.float64)
        for col in range(len(NUMERIC)):
            valid = numeric[:, col][np.isfinite(numeric[:, col])]
            self.medians[col] = float(np.median(valid)) if len(valid) else 0.0
        numeric = np.where(np.isfinite(numeric), numeric, self.medians)
        self.means = numeric.mean(axis=0)
        self.scales = numeric.std(axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        self.feature_names = ["%s=%s" % (name, value)
                              for name in CATEGORICAL
                              for value in self.categories[name]] + list(NUMERIC)
        return self

    def transform(self, rows):
        features = [_features(row) for row in rows]
        width = len(self.feature_names)
        result = np.zeros((len(features), width), dtype=np.float32)
        offset = 0
        for name in CATEGORICAL:
            mapping = {value: index for index, value in enumerate(self.categories[name])}
            unknown_index = mapping[UNKNOWN]
            for i, row in enumerate(features):
                result[i, offset + mapping.get(row[name], unknown_index)] = 1.0
            offset += len(mapping)
        numeric = np.asarray([[row[name] for name in NUMERIC]
                              for row in features], dtype=np.float64)
        numeric = np.where(np.isfinite(numeric), numeric, self.medians)
        result[:, offset:] = ((numeric - self.means) / self.scales).astype(np.float32)
        if result.shape != (len(rows), width) or not np.isfinite(result).all():
            raise ValueError("la matriz de características tiene forma o valores inválidos")
        return np.ascontiguousarray(result)

    def to_dict(self):
        return {
            "categorical": list(CATEGORICAL),
            "numeric": list(NUMERIC),
            "categories": self.categories,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "feature_names": self.feature_names,
        }


def _read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("%s no contiene columnas: %s" %
                             (path, ", ".join(sorted(missing))))
        rows = list(reader)
    if not rows:
        raise ValueError("%s no contiene filas" % path)
    return rows


def _stratified_split(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("cada clase necesita al menos dos filas")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(train), np.asarray(validation)


def _new_classifier(input_size, args):
    model = tensorc.Sequential([
        tensorc.DenseLayer(input_size, 64, seed=args.seed),
        tensorc.ReLULayer(),
        tensorc.DenseLayer(64, 32, seed=args.seed + 1),
        tensorc.ReLULayer(),
        tensorc.DenseLayer(32, 2, seed=args.seed + 2),
    ])
    return tensorc.TensorCClassifier(
        model=model, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, random_state=args.seed, verbose=True,
        log_every=max(1, args.epochs // 10),
    )


def _write_submission(path, ids, predictions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("PassengerId", "Transported"))
        writer.writerows((passenger_id, "True" if bool(label) else "False")
                         for passenger_id, label in zip(ids, predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data",
                        help="directorio con train.csv y test.csv descargados de Kaggle")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7,
                        help="semilla de inicialización y shuffle del modelo")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="semilla fija para la partición de validación")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("--epochs, --batch-size y --learning-rate deben ser positivos")
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("--validation-fraction debe estar entre 0.05 y 0.5")

    train_rows = _read_csv(args.data_dir / "train.csv", ("PassengerId", "Transported"))
    test_rows = _read_csv(args.data_dir / "test.csv", ("PassengerId",))
    labels = np.asarray([row["Transported"].strip().lower() == "true"
                         for row in train_rows], dtype=np.bool_)
    if any(row["Transported"].strip().lower() not in {"true", "false"}
           for row in train_rows):
        raise ValueError("Transported debe contener solamente True o False")

    train_ids, validation_ids = _stratified_split(
        labels, args.validation_fraction, args.split_seed)
    y_train, y_validation = labels[train_ids], labels[validation_ids]
    validation_rows = [train_rows[i] for i in train_ids]
    heldout_rows = [train_rows[i] for i in validation_ids]
    validation_encoder = FeatureEncoder().fit(validation_rows)
    x_train = validation_encoder.transform(validation_rows)
    x_validation = validation_encoder.transform(heldout_rows)

    validation_model = _new_classifier(x_train.shape[1], args)
    start = time.perf_counter()
    validation_model.fit(x_train, y_train)
    validation_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    validation_predictions = validation_model.predict(x_validation)
    validation_seconds = time.perf_counter() - start
    validation_accuracy = float(np.mean(validation_predictions == y_validation))
    baseline_accuracy = float(max(np.mean(y_validation), 1.0 - np.mean(y_validation)))
    confusion = [[int(np.sum((y_validation == actual) &
                             (validation_predictions == predicted)))
                 for predicted in (False, True)] for actual in (False, True)]
    print("Validación: accuracy %.4f (baseline mayoritaria %.4f), %.2f s" %
          (validation_accuracy, baseline_accuracy, validation_seconds))
    print("Matriz de confusión [False, True]: %s" % confusion)

    full_encoder = FeatureEncoder().fit(train_rows)
    x_full = full_encoder.transform(train_rows)
    x_test = full_encoder.transform(test_rows)
    final_model = _new_classifier(x_full.shape[1], args)
    start = time.perf_counter()
    final_model.fit(x_full, labels)
    final_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test_predictions = final_model.predict(x_test)
    prediction_seconds = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "spaceship_titanic.tck"
    final_model.save(model_path)
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["preprocessing"] = full_encoder.to_dict()
    metadata["model"] = {"hidden_layers": [64, 32], "activation": "ReLU"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    submission_path = args.output_dir / "submission.csv"
    _write_submission(submission_path,
                      [row["PassengerId"] for row in test_rows], test_predictions)
    metrics = {
        "validation_accuracy": validation_accuracy,
        "majority_baseline_accuracy": baseline_accuracy,
        "confusion_matrix_labels_false_true": confusion,
        "validation_training_seconds": validation_train_seconds,
        "validation_prediction_seconds": validation_seconds,
        "final_training_seconds": final_train_seconds,
        "test_prediction_seconds": prediction_seconds,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_ids),
        "test_rows": len(test_rows),
        "features": x_full.shape[1],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "tensorc_version": tensorc.__version__,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("Modelo: %s" % model_path)
    print("Envío:  %s" % submission_path)
    print("Métricas: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
