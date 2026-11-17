"""
Course Lab R33-T004: Metric Correctness, Explicit Averaging, and Zero-Denominator Handling.

Scope: 06 decision trees / Regularization Hyperparameters evaluation metric verification.
Note: Verification fixture rows are synthetic (8 rows with x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]).
They are strictly intended for metric validation and pipeline verification.
They are NOT observations from any real dataset and must not be used for performance claims.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import json
import math
from pathlib import Path


@dataclass
class ClassificationMetrics:
    """Detailed classification metrics with explicit confusion counts."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    tn: int
    fn: int
    zero_division_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "zero_division_applied": self.zero_division_applied,
        }


@dataclass
class RegressionMetrics:
    """Detailed regression metrics."""
    mae: float
    mse: float
    rmse: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
        }


def compute_binary_confusion_matrix(
    y_true: List[int], y_pred: List[int], pos_label: int = 1
) -> Tuple[int, int, int, int]:
    """
    Computes (TP, FP, TN, FN) for binary classification.
    Raises ValueError if lengths mismatch or inputs are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")
    if not y_true:
        raise ValueError("Inputs cannot be empty.")

    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == pos_label and yp == pos_label:
            tp += 1
        elif yt != pos_label and yp == pos_label:
            fp += 1
        elif yt != pos_label and yp != pos_label:
            tn += 1
        elif yt == pos_label and yp != pos_label:
            fn += 1

    return tp, fp, tn, fn


def calculate_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> ClassificationMetrics:
    """
    Computes Precision, Recall, F1, and Accuracy for binary classification.
    Explicit zero-denominator behavior:
      - Precision: if (TP + FP) == 0, returns zero_division and flags zero_division_applied=True.
      - Recall: if (TP + FN) == 0, returns zero_division and flags zero_division_applied=True.
      - F1: if (precision + recall) == 0, returns zero_division.
    """
    tp, fp, tn, fn = compute_binary_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else zero_division

    zero_div_flag = False

    # Precision = TP / (TP + FP)
    if (tp + fp) == 0:
        precision = zero_division
        zero_div_flag = True
    else:
        precision = tp / (tp + fp)

    # Recall = TP / (TP + FN)
    if (tp + fn) == 0:
        recall = zero_division
        zero_div_flag = True
    else:
        recall = tp / (tp + fn)

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) == 0:
        f1_score = zero_division
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        zero_division_applied=zero_div_flag,
    )


def calculate_regression_metrics(y_true: List[float], y_pred: List[float]) -> RegressionMetrics:
    """
    Computes MAE, MSE, and RMSE for continuous targets.
    MAE = mean(|y_true - y_pred|)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")
    if not y_true:
        raise ValueError("Inputs cannot be empty.")

    n = len(y_true)
    abs_errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
    sq_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]

    mae = sum(abs_errors) / n
    mse = sum(sq_errors) / n
    rmse = math.sqrt(mse)

    return RegressionMetrics(mae=mae, mse=mse, rmse=rmse)


class DecisionStumpClassifier:
    """
    Single-threshold regularized decision tree stump:
    Predicts label 1 if feature >= threshold, else 0.
    """

    def __init__(self, feature_key: str = "x1", threshold: float = 3.5) -> None:
        self.feature_key = feature_key
        self.threshold = threshold

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        return [
            1 if r[self.feature_key] >= self.threshold else 0
            for r in records
        ]

    def evaluate(self, records: List[Dict[str, Any]]) -> ClassificationMetrics:
        y_true = [r["label"] for r in records]
        y_pred = self.predict(records)
        return calculate_classification_metrics(y_true, y_pred)


def fixed_stratified_split(
    records: List[Dict[str, Any]], train_ratio: float = 0.5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic stratified split by label preserving class balance."""
    by_label: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    train_set, test_set = [], []
    for lbl in sorted(by_label.keys()):
        group = by_label[lbl]
        n_train = max(1, round(len(group) * train_ratio))
        train_set.extend(group[:n_train])
        test_set.extend(group[n_train:])

    return train_set, test_set


def load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Loads records from JSON or CSV file, or returns in-memory list."""
    if isinstance(source, list):
        return source

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON must be a list of records.")
            return data

    if p.suffix.lower() == ".csv":
        records = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "record_id": row["record_id"].strip(),
                    "group_id": row["group_id"].strip(),
                    "x1": float(row["x1"].strip()) if "." in row["x1"] else int(row["x1"].strip()),
                    "x2": float(row["x2"].strip()) if "." in row["x2"] else int(row["x2"].strip()),
                    "label": int(row["label"].strip()),
                })
        return records

    raise ValueError(f"Unsupported format: {p.suffix}")
