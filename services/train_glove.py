"""Train a RandomForest glove classifier from CSV sensor samples."""

from __future__ import annotations

import csv
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "euler_h",
    "euler_r",
    "euler_p",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "laccel_x",
    "laccel_y",
    "laccel_z",
]


def load_dataset(csv_path: Path) -> tuple[list[list[float]], list[str]]:
    """Load glove training features and labels from a CSV file."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    features: list[list[float]] = []
    labels: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [column for column in ["label", *FEATURE_COLUMNS] if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        for row in reader:
            label = str(row["label"]).strip().upper()
            if not label:
                continue
            labels.append(label)
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
    if not features:
        raise ValueError("Dataset is empty.")
    return features, labels


def train_glove_model(
    csv_path: Path = Path("speak_sign_app/data/glove_samples.csv"),
    model_path: Path = Path("speak_sign_app/assets/models/glove_rf.pkl"),
) -> Path:
    """Train and save the glove RandomForest classifier."""

    features, labels = load_dataset(csv_path)
    stratify = labels if len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    classifier = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(x_train, y_train)
    print(classification_report(y_test, classifier.predict(x_test), zero_division=0))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(classifier, file)
    print(f"Saved {model_path}")
    return model_path


if __name__ == "__main__":
    train_glove_model()


__all__ = ["FEATURE_COLUMNS", "load_dataset", "train_glove_model"]
