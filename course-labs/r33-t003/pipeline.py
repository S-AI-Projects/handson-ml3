"""
Course Lab R33-T003: Minimal Majority-Class Baseline for Image Classification.

Scope: 19 training and deploying at scale / Exporting SavedModels image contract.
Note: Verification fixture rows are synthetic 4x4 grayscale images (three all-zero,
three white diagonals). This is a CPU smoke fixture, NOT the named real dataset.
It must not be used as evidence of real-world or clinical predictive performance.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import json
from pathlib import Path


class SchemaError(Exception):
    """Raised when record data fails the image classification contract."""
    pass


class NotFittedError(Exception):
    """Raised when predict/evaluate is called before fit."""
    pass


@dataclass
class BaselineEvaluation:
    """Holds evaluation results for the baseline model."""
    majority_class: int
    train_sample_count: int
    train_class_counts: Dict[int, int]
    test_sample_count: int
    test_class_counts: Dict[int, int]
    correct_predictions: int
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "majority_class": self.majority_class,
            "train_sample_count": self.train_sample_count,
            "train_class_counts": self.train_class_counts,
            "test_sample_count": self.test_sample_count,
            "test_class_counts": self.test_class_counts,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
        }


def validate_image_record(record: Dict[str, Any], row_idx: int = 0) -> Dict[str, Any]:
    """
    Validates a single image classification record against the 4x4 contract:
      - record_id: unique non-empty string or int.
      - group_id: non-empty string or int.
      - pixels: list of 16 values in [0, 255].
      - label: integer in {0, 1}.
    """
    rec_id = record.get("record_id")
    if rec_id is None or str(rec_id).strip() == "":
        raise SchemaError(f"Row {row_idx}: missing or empty 'record_id'")

    grp_id = record.get("group_id")
    if grp_id is None or str(grp_id).strip() == "":
        raise SchemaError(f"Record '{rec_id}': missing or empty 'group_id'")

    pixels = record.get("pixels")
    if not isinstance(pixels, list) or len(pixels) != 16:
        raise SchemaError(f"Record '{rec_id}': 'pixels' must be a list of 16 values, got {type(pixels).__name__} len={len(pixels) if isinstance(pixels, list) else 'N/A'}")

    for p_idx, px in enumerate(pixels):
        if isinstance(px, bool) or not isinstance(px, (int, float)) or px < 0 or px > 255:
            raise SchemaError(f"Record '{rec_id}': pixel {p_idx} must be numeric in [0, 255], got {px!r}")

    label = record.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
        raise SchemaError(f"Record '{rec_id}': 'label' must be binary integer in {{0, 1}}, got {label!r}")

    return record


def validate_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validates all image classification records and ensures uniqueness of record_id."""
    if not records:
        raise SchemaError("Dataset cannot be empty.")

    seen_ids = set()
    for idx, r in enumerate(records):
        validate_image_record(r, idx)
        rec_id = str(r["record_id"]).strip()
        if rec_id in seen_ids:
            raise SchemaError(f"Duplicate record_id detected: {rec_id!r}")
        seen_ids.add(rec_id)

    return records


class MajorityClassBaseline:
    """
    Simplest honest baseline for image classification:
    Fits strictly on training labels, learning the single most frequent class.
    Predicts this majority class constant for any unseen held-out sample.
    """

    def __init__(self) -> None:
        self.majority_class: Optional[int] = None
        self.train_class_counts: Dict[int, int] = {}
        self.train_sample_count: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.majority_class is not None

    def fit(self, train_records: List[Dict[str, Any]]) -> "MajorityClassBaseline":
        """Learns the majority class using ONLY training observations."""
        validate_dataset(train_records)

        labels = [r["label"] for r in train_records]
        if not labels:
            raise ValueError("Training dataset cannot be empty.")

        counts = Counter(labels)
        # Sort by count desc, then label asc for strict determinism on ties
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        self.majority_class = sorted_counts[0][0]
        self.train_class_counts = dict(counts)
        self.train_sample_count = len(labels)
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """Predicts the learned majority class for each input observation."""
        if not self.is_fitted:
            raise NotFittedError("Cannot predict before fitting on training data.")
        validate_dataset(records)
        return [self.majority_class for _ in records]

    def evaluate(self, test_records: List[Dict[str, Any]]) -> BaselineEvaluation:
        """Evaluates baseline accuracy on held-out test data."""
        if not self.is_fitted:
            raise NotFittedError("Cannot evaluate before fitting on training data.")

        validate_dataset(test_records)
        y_true = [r["label"] for r in test_records]
        y_pred = self.predict(test_records)

        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = correct / len(y_true) if y_true else 0.0

        return BaselineEvaluation(
            majority_class=self.majority_class,
            train_sample_count=self.train_sample_count,
            train_class_counts=self.train_class_counts,
            test_sample_count=len(y_true),
            test_class_counts=dict(Counter(y_true)),
            correct_predictions=correct,
            accuracy=accuracy,
        )


def split_train_test(
    records: List[Dict[str, Any]],
    train_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically partitions records into train and test sets by record_id.
    Default benchmark partition for 6-row fixture:
      - Train: img_01, img_02, img_03, img_04 (3 zeroes, 1 diagonal -> majority is 0)
      - Test: img_05, img_06 (2 diagonals -> held-out evaluation)
    """
    validate_dataset(records)
    record_map = {str(r["record_id"]).strip(): r for r in records}

    if train_ids is None:
        train_ids = ["img_01", "img_02", "img_03", "img_04"]
    if test_ids is None:
        test_ids = ["img_05", "img_06"]

    train_set_ids = set(train_ids)
    test_set_ids = set(test_ids)

    # Disjointness check
    overlap = train_set_ids & test_set_ids
    if overlap:
        raise ValueError(f"Train and test sets must be disjoint. Overlapping IDs: {overlap}")

    train_records = [record_map[i] for i in train_ids]
    test_records = [record_map[i] for i in test_ids]

    return train_records, test_records


def load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Loads records from a JSON or CSV file, or returns in-memory list."""
    if isinstance(source, list):
        return source

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a list of record objects.")
            return data

    if p.suffix.lower() == ".csv":
        records = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pixels_raw = row["pixels"].strip()
                if pixels_raw.startswith("[") and pixels_raw.endswith("]"):
                    pixels = json.loads(pixels_raw)
                else:
                    pixels = [int(v.strip()) for v in pixels_raw.split(",")]

                records.append({
                    "record_id": row["record_id"].strip(),
                    "group_id": row["group_id"].strip(),
                    "label": int(row["label"].strip()),
                    "pixels": pixels,
                })
        return records

    raise ValueError(f"Unsupported format: {p.suffix}")
