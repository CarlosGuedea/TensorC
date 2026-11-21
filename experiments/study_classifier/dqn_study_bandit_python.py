#!/usr/bin/env python3
"""Python/NumPy DQN contextual-bandit example for study-place images."""

import argparse
import csv
import json
import random
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

import tensorc

IMAGE_SIZE = 32
HOG_BINS = 9
FEATURE_COUNT = 359
EPISODES = 8000
SEED = 42


def split_rows(rows, seed=SEED):
    rng = random.Random(seed)
    by_label = {"apto": [], "no_apto": []}
    for row in rows:
        if row["study_label"] in by_label and row.get("foto_local"):
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
    uncertain = [
        row for row in rows
        if row["study_label"] == "incierto" and row.get("foto_local")
    ]
    return splits, uncertain


def metadata_features(row):
    occupancy = {
        "despejado": [1, 0, 0], "medio": [0, 1, 0], "lleno": [0, 0, 1],
    }.get(row["visual_occupancy"], [0, 0, 0])
    space = {
        "exterior": [1, 0, 0], "interior": [0, 1, 0], "mixto": [0, 0, 1],
    }.get(row["space_type"], [0, 0, 0])
    values = [
        min(float(row["people_count"] or 0), 30.0) / 30.0,
        float(row["has_table"] == "True"),
        *occupancy, *space,
        (float(row["temperature_c"] or 30) - 20.0) / 20.0,
        float(row["humidity_pct"] or 0) / 100.0,
        (float(row["sound_dbfs"] or -40) + 40.0) / 40.0,
    ]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (11,) or not np.isfinite(result).all():
        raise ValueError("Metadatos inválidos para id=%s" % row.get("id", "?"))
    return result


def hog_features(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    gy[0, :] = gray[1, :] - gray[0, :]
    gy[-1, :] = gray[-1, :] - gray[-2, :]

    magnitude = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    bin_index = np.minimum((angle * HOG_BINS / 180.0).astype(np.int32),
                           HOG_BINS - 1)
    cells = np.zeros((4, 4, HOG_BINS), dtype=np.float32)
    for cell_y in range(4):
        for cell_x in range(4):
            y0, x0 = cell_y * 8, cell_x * 8
            bins = bin_index[y0:y0 + 8, x0:x0 + 8].ravel()
            weights = magnitude[y0:y0 + 8, x0:x0 + 8].ravel()
            cells[cell_y, cell_x] = np.bincount(
                bins, weights=weights, minlength=HOG_BINS
            )
    blocks = []
    for block_y in range(3):
        for block_x in range(3):
            block = cells[block_y:block_y + 2, block_x:block_x + 2].ravel()
            block = block / np.sqrt(np.dot(block, block) + 1e-6)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(np.dot(block, block) + 1e-6)
            blocks.extend(block)
    return np.asarray(blocks, dtype=np.float32)


def image_features(image, metadata):
    image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    rgb = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    hog = hog_features(gray)
    color = np.concatenate([
        np.histogram(rgb[:, :, channel], bins=8, range=(0, 256))[0]
        for channel in range(3)
    ]).astype(np.float32) / float(IMAGE_SIZE * IMAGE_SIZE)
    result = np.concatenate([hog, color, metadata_features(metadata)])
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError("Vector de características inválido")
    return result


def photo_member(prefix, row):
    relative = PurePosixPath(row["foto_local"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Ruta de foto inválida: %s" % row["foto_local"])
    return str(PurePosixPath(prefix) / relative)


def load_data(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        csv_member = next(
            (name for name in members
             if name.endswith("/datos.csv") or name == "datos.csv"),
            None,
        )
        if csv_member is None:
            raise ValueError("El ZIP no contiene datos.csv")
        prefix = csv_member.rsplit("/", 1)[0] if "/" in csv_member else ""
        with archive.open(csv_member) as raw:
            rows = list(csv.DictReader(
                (line.decode("utf-8-sig") for line in raw)
            ))
        required = {
            "id", "study_label", "people_count", "has_table",
            "visual_occupancy", "space_type", "temperature_c",
            "humidity_pct", "sound_dbfs", "foto_local",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError("datos.csv está vacío o le faltan columnas requeridas")
        splits, uncertain = split_rows(rows)
        uncertain_count = sum(row["study_label"] == "incierto" for row in rows)
        features = {}
        for split_name in ("train", "test"):
            for row in splits[split_name]:
                member = photo_member(prefix, row)
                if member not in members:
                    raise FileNotFoundError("Falta foto del split %s: %s" %
                                            (split_name, member))
                with archive.open(member) as image_file:
                    with Image.open(image_file) as image:
                        features[row["id"]] = image_features(image, row)
        available_uncertain = []
        for row in uncertain:
            member = photo_member(prefix, row)
            if member not in members:
                continue
            with archive.open(member) as image_file:
                with Image.open(image_file) as image:
                    features[row["id"]] = image_features(image, row)
            available_uncertain.append(row)
    return splits, available_uncertain, features, uncertain_count


class StudyBanditEnv:
    """Entorno Gymnasium-style de un paso, con recompensa de la etiqueta."""

    def __init__(self, rows, features):
        self.rows = rows
        self.features = features
        self.rng = np.random.default_rng(SEED)
        self.current = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current = self.rows[int(self.rng.integers(len(self.rows)))]
        return self.features[self.current["id"]], {}

    def step(self, action):
        if self.current is None:
            raise RuntimeError("llama reset antes de step")
        if action not in (0, 1):
            raise ValueError("acción debe ser 0=no_apto o 1=apto")
        label = int(self.current["study_label"] == "apto")
        reward = 1.0 if action == label else -1.0
        return self.features[self.current["id"]], reward, True, False, {}


def binary_metrics(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_apto": float(precision),
        "recall_apto": float(recall),
        "f1_apto": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "test_rows": int(len(y_true)),
    }


def predict_rows(agent, rows, features):
    return np.asarray([
        agent.select_action(features[row["id"]], epsilon=0.0) for row in rows
    ], dtype=np.int32)


def save_predictions(path, rows, predictions):
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "real", "prediccion"])
        for row, prediction in zip(rows, predictions):
            label = ("apto" if prediction else "no_apto")
            writer.writerow([row["id"], row["study_label"], label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="ZIP con datos.csv y fotos/")
    parser.add_argument("--output", default="experiments/study_classifier/results/python_dqn")
    parser.add_argument("--photo", help="Foto adicional para inferencia tras entrenar")
    parser.add_argument("--temperature-c", type=float)
    parser.add_argument("--humidity-pct", type=float, default=0.0)
    parser.add_argument("--people", type=float)
    parser.add_argument("--has-table", choices=("true", "false"))
    parser.add_argument("--occupancy", choices=("despejado", "medio", "lleno"))
    parser.add_argument("--space-type", choices=("exterior", "interior", "mixto"))
    parser.add_argument("--sound-dbfs", type=float)
    args = parser.parse_args()

    if args.photo and any(value is None for value in (
        args.temperature_c, args.people, args.has_table, args.occupancy,
        args.space_type, args.sound_dbfs,
    )):
        parser.error("--photo requiere temperatura, personas, mesa, ocupación, "
                     "tipo de espacio y ruido dBFS")

    feature_started = time.perf_counter()
    splits, uncertain_rows, features, uncertain_count = load_data(args.zip)
    feature_seconds = time.perf_counter() - feature_started
    train_rows = splits["train"]
    test_rows = splits["test"]
    env = StudyBanditEnv(train_rows, features)
    agent = tensorc.DQNAgent(
        FEATURE_COUNT, 2, hidden_size=64, replay_capacity=1024,
        batch_size=32, target_update_interval=100,
        learning_rate=0.0005, discount_factor=0.0, seed=SEED,
    )

    train_started = time.perf_counter()
    training_hits = 0
    for episode in range(EPISODES):
        observation, _ = env.reset(seed=SEED + episode)
        epsilon = 1.0 - 0.95 * episode / (EPISODES - 1)
        action = agent.select_action(observation, epsilon=epsilon)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        agent.remember(observation, action, reward, next_observation,
                       terminated=terminated, truncated=truncated)
        training_hits += reward > 0.0
        agent.train_step()
    train_seconds = time.perf_counter() - train_started

    predictions = predict_rows(agent, test_rows, features)
    labels = np.asarray([row["study_label"] == "apto" for row in test_rows],
                        dtype=np.int32)
    metrics = binary_metrics(labels, predictions)
    metrics.update({
        "train_rows": len(train_rows),
        "uncertain_rows_excluded": uncertain_count,
        "uncertain_photos_available": len(uncertain_rows),
        "feature_count": FEATURE_COUNT,
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": train_seconds,
        "episodes": EPISODES,
    })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "policy.npz"
    agent.save(policy_path)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    save_predictions(output / "test_predictions.csv", test_rows, predictions)
    uncertain_predictions = predict_rows(agent, uncertain_rows, features)
    save_predictions(output / "uncertain_predictions.csv", uncertain_rows,
                     uncertain_predictions)
    reloaded = tensorc.DQNAgent.load(policy_path)
    if not np.array_equal(predictions, predict_rows(reloaded, test_rows, features)):
        raise RuntimeError("el checkpoint recargado cambió las predicciones")

    print("split: %d train, %d test; inciertas disponibles: %d" %
          (len(train_rows), len(test_rows), len(uncertain_rows)))
    print("características: %d | extracción: %.3f s" %
          (FEATURE_COUNT, feature_seconds))
    print("entrenamiento DQN: %d episodios | %.3f s | "
          "aciertos explorando %.2f%%" %
          (EPISODES, train_seconds, 100.0 * training_hits / EPISODES))
    print(json.dumps(metrics, indent=2))
    print("checkpoint: %s" % policy_path.resolve())

    if args.photo:
        custom_meta = {
            "id": "custom", "people_count": str(args.people),
            "has_table": "True" if args.has_table == "true" else "False",
            "visual_occupancy": args.occupancy,
            "space_type": args.space_type,
            "temperature_c": str(args.temperature_c),
            "humidity_pct": str(args.humidity_pct),
            "sound_dbfs": str(args.sound_dbfs),
        }
        with Image.open(args.photo) as image:
            state = image_features(image, custom_meta)
        prediction = agent.select_action(state, epsilon=0.0)
        if reloaded.select_action(state, epsilon=0.0) != prediction:
            raise RuntimeError("el checkpoint recargado cambió la inferencia")
        print("foto nueva (Python DQN): %s" %
              ("apto" if prediction else "no_apto"))


if __name__ == "__main__":
    main()
