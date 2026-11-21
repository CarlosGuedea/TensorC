#!/usr/bin/env python3
"""Entrenamiento multimodal de lugares para estudiar con TensorC.

TensorC ejecuta las operaciones de la red y el autograd; Python solo carga
el ZIP, prepara las imágenes/metadatos y calcula las métricas.
"""

import argparse
import csv
import json
import random
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

import tensorc


def metadata(row):
    occupancy = {"despejado": [1, 0, 0], "medio": [0, 1, 0],
                 "lleno": [0, 0, 1]}.get(row["visual_occupancy"], [0, 0, 0])
    space = {"exterior": [1, 0, 0], "interior": [0, 1, 0],
             "mixto": [0, 0, 1]}.get(row["space_type"], [0, 0, 0])
    temperature = float(row["temperature_c"] or 30)
    humidity = float(row["humidity_pct"] or 0)
    sound = float(row["sound_dbfs"] or -40)
    return np.asarray([
        min(float(row["people_count"] or 0), 30) / 30,
        float(row["has_table"] == "True"),
        *occupancy, *space,
        (temperature - 20) / 20,
        humidity / 100,
        (sound + 40) / 40,
    ], dtype=np.float32)


def read_image(zf, row, prefix):
    with zf.open(prefix + row["foto_local"]) as f:
        image = Image.open(BytesIO(f.read())).convert("RGB")
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255


def split_rows(rows, seed):
    rng = random.Random(seed)
    by_label = {"apto": [], "no_apto": []}
    for row in rows:
        if row["study_label"] in by_label and row["foto_local"]:
            by_label[row["study_label"]].append(row)
    for values in by_label.values():
        rng.shuffle(values)
    splits = {"train": [], "val": [], "test": []}
    for values in by_label.values():
        n_train = int(len(values) * 0.70)
        n_val = int(len(values) * 0.15)
        splits["train"] += values[:n_train]
        splits["val"] += values[n_train:n_train + n_val]
        splits["test"] += values[n_train + n_val:]
    for values in splits.values():
        rng.shuffle(values)
    uncertain = [r for r in rows if r["study_label"] == "incierto" and r["foto_local"]]
    return splits, uncertain


def load_arrays(zf, rows, prefix, unknown=False):
    images = np.ascontiguousarray(
        np.stack([read_image(zf, r, prefix) for r in rows]), dtype=np.float32)
    features = np.ascontiguousarray(np.stack([metadata(r) for r in rows]), dtype=np.float32)
    labels = np.asarray(
        [255 if unknown else int(r["study_label"] == "apto") for r in rows],
        dtype=np.int32,
    )
    return images, features, labels


class StudyNetwork:
    def __init__(self):
        self.conv1 = tensorc.Conv2DLayer(3, 8, 3, 3, padding=(1, 1), seed=101)
        self.conv2 = tensorc.Conv2DLayer(8, 16, 3, 3, padding=(1, 1), seed=202)
        self.image_projection = tensorc.DenseLayer(1024, 16, seed=303)
        self.metadata_projection = tensorc.DenseLayer(11, 8, seed=404)
        self.classifier = tensorc.DenseLayer(24, 2, seed=505)
        self.params = [
            self.conv1.kernel, self.conv1.bias,
            self.conv2.kernel, self.conv2.bias,
            self.image_projection.weight, self.image_projection.bias,
            self.metadata_projection.weight, self.metadata_projection.bias,
            self.classifier.weight, self.classifier.bias,
        ]
        self.optimizer = tensorc.Adam(self.params, 0.001)

    def forward(self, images, features):
        x = self.conv1.forward(images).relu().maxpool2d(2, 2)
        x = self.conv2.forward(x).relu().maxpool2d(2, 2)
        x = x.reshape((int(x.shape[0]), 1024))
        visual = self.image_projection.forward(x).relu()
        tabular = self.metadata_projection.forward(features).relu()
        joined = visual.concat(tabular, axis=1)
        return self.classifier.forward(joined)

    def save(self, path):
        tensorc.save(self.params, str(path))

    def load(self, path):
        tensors = tensorc.load(str(path), requires_grad=True)
        for parameter, saved in zip(self.params, tensors):
            parameter.copy_(saved)


def tensor_batch(array):
    return tensorc.Tensor(np.ascontiguousarray(array, dtype=np.float32))


def one_hot(labels):
    target = np.zeros((len(labels), 2), dtype=np.float32)
    target[np.arange(len(labels)), labels] = 1
    return tensor_batch(target)


def balanced_order(labels, seed):
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    size = max(len(negative), len(positive))
    order = np.concatenate([
        rng.choice(negative, size=size, replace=True),
        rng.choice(positive, size=size, replace=True),
    ])
    rng.shuffle(order)
    return order


def probabilities(model, images, features, batch_size=16):
    output = []
    for start in range(0, len(images), batch_size):
        end = start + batch_size
        probs = model.forward(tensor_batch(images[start:end]),
                              tensor_batch(features[start:end])).softmax()
        output.extend(np.asarray(probs.data, dtype=np.float32).reshape(-1, 2)[:, 1])
    return np.asarray(output, dtype=np.float32)


def metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int32)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    auc = float((positive[:, None] > negative[None, :]).mean() +
                0.5 * (positive[:, None] == negative[None, :]).mean())
    return {
        "accuracy": float((predictions == labels).mean()),
        "precision_apto": float(precision),
        "recall_apto": float(recall),
        "f1_apto": float(f1),
        "roc_auc": auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def write_predictions(path, rows, probabilities, unknown=False):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "id", "real", "prediccion", "prob_apto"])
        for i, (row, probability) in enumerate(zip(rows, probabilities)):
            writer.writerow([
                i, row["id"], "incierto" if unknown else row["study_label"],
                "apto" if probability >= 0.5 else "no_apto", f"{probability:.7f}",
            ])


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        csv_member = next(name for name in zf.namelist()
                          if name.endswith("/datos.csv") or name == "datos.csv")
        prefix = csv_member[:-len("datos.csv")]
        with zf.open(csv_member) as f:
            rows = list(csv.DictReader((line.decode("utf-8-sig") for line in f)))
        splits, uncertain_rows = split_rows(rows, args.seed)
        data = {name: load_arrays(zf, values, prefix)
                for name, values in splits.items()}
        uncertain = load_arrays(zf, uncertain_rows, prefix, unknown=True)

    model = StudyNetwork()
    best_loss = float("inf")
    history = []
    best_path = output / "study_classifier_python.tck"

    for epoch in range(1, args.epochs + 1):
        images, features, labels = data["train"]
        order = balanced_order(labels, args.seed + epoch)
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            logits = model.forward(tensor_batch(images[indices]),
                                   tensor_batch(features[indices]))
            loss = logits.softmax().cross_entropy(one_hot(labels[indices]))
            loss.backward()
            model.optimizer.step()
            losses.append(float(loss.data[0]))

        val_images, val_features, val_labels = data["val"]
        val_probabilities = probabilities(model, val_images, val_features, args.batch_size)
        val_metrics = metrics(val_labels, val_probabilities)
        val_loss = float(-np.mean(
            val_labels * np.log(np.maximum(val_probabilities, 1e-7)) +
            (1 - val_labels) * np.log(np.maximum(1 - val_probabilities, 1e-7))))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, **val_metrics})
        print(f"epoca {epoch:02d}/{args.epochs} | train_loss {np.mean(losses):.5f} | "
              f"val_loss {val_loss:.5f} | val_acc {val_metrics['accuracy']:.2%}")
        if val_loss < best_loss:
            best_loss = val_loss
            model.save(best_path)

    model.load(best_path)
    test_images, test_features, test_labels = data["test"]
    test_probabilities = probabilities(model, test_images, test_features, args.batch_size)
    test_metrics = metrics(test_labels, test_probabilities)
    write_predictions(output / "test_predictions_python.csv", splits["test"], test_probabilities)
    uncertain_probabilities = probabilities(model, uncertain[0], uncertain[1], args.batch_size)
    write_predictions(output / "uncertain_predictions_python.csv", uncertain_rows,
                      uncertain_probabilities, unknown=True)
    (output / "history_python.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "metrics_python.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("\nPRUEBA:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="ZIP con datos.csv y fotos/")
    parser.add_argument("--output", default="results_python")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
