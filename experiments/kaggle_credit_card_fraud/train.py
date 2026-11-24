#!/usr/bin/env python3
"""Train a TensorC model for Kaggle's Predicting Credit Card Fraud."""

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorc


NUMERIC = (
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score",
    "prior_disputes",
)
CATEGORICAL = (
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
MISSING = "__MISSING__"
UNKNOWN = "__UNKNOWN__"
REQUIRED_TRAIN = {"id", "is_fraud", *NUMERIC, *CATEGORICAL}


def _read_csv(path, required):
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("%s omite columnas: %s" % (path, ", ".join(sorted(missing))))
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("%s debe tener filas e identificadores únicos" % path)
    return frame


def _category_values(series):
    return series.astype("string").fillna(MISSING).astype(str).to_numpy()


class FeatureEncoder:
    """Impute/scale numeric columns and one-hot encode low-cardinality fields."""

    def fit(self, frame):
        self.categories = {}
        for name in CATEGORICAL:
            values = _category_values(frame[name])
            self.categories[name] = sorted(set(values) - {MISSING}) + [MISSING, UNKNOWN]
        numeric = frame.loc[:, NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64)
        self.medians = np.zeros(len(NUMERIC), dtype=np.float64)
        for column in range(len(NUMERIC)):
            valid = numeric[:, column][np.isfinite(numeric[:, column])]
            self.medians[column] = np.median(valid) if len(valid) else 0.0
        numeric = np.where(np.isfinite(numeric), numeric, self.medians)
        self.means = numeric.mean(axis=0)
        self.scales = numeric.std(axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        self.feature_names = [
            "%s=%s" % (name, value)
            for name in CATEGORICAL for value in self.categories[name]
        ] + list(NUMERIC)
        return self

    def transform(self, frame):
        result = np.zeros((len(frame), len(self.feature_names)), dtype=np.float32)
        row_indices = np.arange(len(frame))
        offset = 0
        for name in CATEGORICAL:
            categories = self.categories[name]
            mapping = {value: index for index, value in enumerate(categories)}
            unknown = mapping[UNKNOWN]
            codes = np.fromiter(
                (mapping.get(value, unknown) for value in _category_values(frame[name])),
                dtype=np.int32, count=len(frame))
            result[row_indices, offset + codes] = 1.0
            offset += len(categories)
        numeric = frame.loc[:, NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64)
        numeric = np.where(np.isfinite(numeric), numeric, self.medians)
        result[:, offset:] = ((numeric - self.means) / self.scales).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("la matriz de características contiene valores no finitos")
        return np.ascontiguousarray(result)

    def to_dict(self):
        return {
            "numeric": list(NUMERIC), "categorical": list(CATEGORICAL),
            "categories": self.categories, "medians": self.medians.tolist(),
            "means": self.means.tolist(), "scales": self.scales.tolist(),
            "feature_names": self.feature_names,
        }


def _stratified_split(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("se requieren al menos dos ejemplos de cada clase")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    train, validation = np.asarray(train), np.asarray(validation)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _oversampled_indices(labels, positive_fraction, seed):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("el entrenamiento necesita ejemplos de ambas clases")
    target_positive = math.ceil(len(negative) * positive_fraction / (1.0 - positive_fraction))
    extra = max(0, target_positive - len(positive))
    rng = np.random.default_rng(seed)
    duplicates = rng.choice(positive, size=extra, replace=True) if extra else np.empty(0, int)
    indices = np.concatenate((np.arange(len(labels)), duplicates))
    rng.shuffle(indices)
    return indices


def _roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("AUC requiere etiquetas y probabilidades finitas del mismo tamaño")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC requiere ejemplos de ambas clases")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[start:end] = (start + 1 + end) / 2.0
    positive_rank_sum = ranks[labels[order]].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def _new_classifier(input_size, args):
    network = tensorc.Sequential([
        tensorc.DenseLayer(input_size, 64, seed=args.seed),
        tensorc.ReLULayer(),
        tensorc.DenseLayer(64, 32, seed=args.seed + 1),
        tensorc.ReLULayer(),
        tensorc.DenseLayer(32, 2, seed=args.seed + 2),
    ])
    return tensorc.TensorCClassifier(
        model=network, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, random_state=args.seed,
        verbose=True, log_every=max(1, args.epochs // 10))


def _write_submission(path, ids, probabilities):
    if len(ids) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("IDs y probabilidades no coinciden o contienen valores inválidos")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "is_fraud"))
        writer.writerows((int(identifier), "%.9g" % float(probability))
                         for identifier, probability in zip(ids, probabilities))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=project / "data")
    parser.add_argument("--output-dir", type=Path, default=project / "artifacts")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--positive-fraction", type=float, default=0.1,
                        help="fracción objetivo de fraudes mediante sobremuestreo")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size y learning-rate deben ser positivos")
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("validation-fraction debe estar entre 0.05 y 0.5")
    if not 0.02 <= args.positive_fraction <= 0.5:
        parser.error("positive-fraction debe estar entre 0.02 y 0.5")

    train_frame = _read_csv(args.data_dir / "train.csv", REQUIRED_TRAIN)
    test_frame = _read_csv(args.data_dir / "test.csv", {"id", *NUMERIC, *CATEGORICAL})
    raw_labels = pd.to_numeric(train_frame["is_fraud"], errors="raise").to_numpy()
    if not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("is_fraud debe contener solamente 0 o 1")
    labels = raw_labels.astype(np.int32)

    fit_indices, validation_indices = _stratified_split(
        labels, args.validation_fraction, args.split_seed)
    encoder = FeatureEncoder().fit(train_frame.iloc[fit_indices])
    x_fit = encoder.transform(train_frame.iloc[fit_indices])
    x_validation = encoder.transform(train_frame.iloc[validation_indices])
    y_fit, y_validation = labels[fit_indices], labels[validation_indices]
    sample_indices = _oversampled_indices(y_fit, args.positive_fraction, args.seed)
    validation_model = _new_classifier(x_fit.shape[1], args)
    start = time.perf_counter()
    validation_model.fit(x_fit[sample_indices], y_fit[sample_indices])
    validation_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    validation_scores = validation_model.predict_proba(x_validation)[:, 1]
    validation_prediction_seconds = time.perf_counter() - start
    auc = _roc_auc(y_validation, validation_scores)
    print("Validación ROC AUC: %.6f (azar 0.500000)" % auc)
    del validation_model, x_fit, x_validation, sample_indices, y_fit
    gc.collect()

    encoder = FeatureEncoder().fit(train_frame)
    x_full = encoder.transform(train_frame)
    x_test = encoder.transform(test_frame)
    sample_indices = _oversampled_indices(labels, args.positive_fraction, args.seed)
    model = _new_classifier(x_full.shape[1], args)
    start = time.perf_counter()
    model.fit(x_full[sample_indices], labels[sample_indices])
    final_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = model.predict_proba(x_test)[:, 1]
    prediction_seconds = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "credit_fraud.tck"
    model.save(model_path)
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["preprocessing"] = encoder.to_dict()
    metadata["model"] = {"hidden_layers": [64, 32], "activation": "ReLU"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_submission(args.output_dir / "submission.csv", test_frame["id"], probabilities)

    metrics = {
        "validation_roc_auc": auc,
        "random_baseline_roc_auc": 0.5,
        "validation_training_seconds": validation_train_seconds,
        "validation_prediction_seconds": validation_prediction_seconds,
        "final_training_seconds": final_train_seconds,
        "test_prediction_seconds": prediction_seconds,
        "train_rows": len(train_frame), "validation_rows": len(validation_indices),
        "test_rows": len(test_frame), "fraud_rows": int(labels.sum()),
        "features": x_full.shape[1], "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "positive_fraction": args.positive_fraction, "seed": args.seed,
        "split_seed": args.split_seed, "tensorc_version": tensorc.__version__,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("Modelo:", model_path)
    print("Envío: ", args.output_dir / "submission.csv")
    print("Métricas:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
