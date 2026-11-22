"""
Course Lab R33-T034: Evaluation Metric Correctness & Text Pipeline Verification.

Student: Snehakumari Vinod Singh
Task: R33-T034 (Task 4 of 10)
Target Item: 16_nlp_with_rnns_and_attention.ipynb (Generating Fake Shakespearean Text)

Disclaimer:
The six fictional text records (t1..t6) and benchmark evaluation fixtures are designed
strictly for CPU smoke testing and metric verification. They are NOT real-world NLP
corpora and must not be used as evidence for real-world or clinical predictive performance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import csv
import json
import math
from pathlib import Path


@dataclass
class ValidationError:
    """Represents a specific schema validation failure for a record."""
    record_id: Any
    field: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field": self.field,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return f"ValidationError(record_id={self.record_id!r}, field={self.field!r}, reason={self.reason!r})"


class SchemaValidator:
    """
    Validates text classification records against the R33-T034 data contract:
      - record_id: Unique string identifier (e.g. t1..t6).
      - text: Non-empty string.
      - label: Binary class identifier {0, 1}.
    """

    def __init__(self, allowed_labels: Optional[Set[int]] = None) -> None:
        self.allowed_labels = allowed_labels if allowed_labels is not None else {0, 1}

    def validate_record(
        self,
        record: Any,
        seen_ids: Optional[Set[Any]] = None,
    ) -> List[ValidationError]:
        if not isinstance(record, dict):
            return [ValidationError(
                record_id=None,
                field="record",
                reason=f"Expected record dict, got {type(record).__name__}.",
            )]

        record_id = record.get("record_id")
        errors: List[ValidationError] = []

        # 1. record_id
        if record_id is None:
            errors.append(ValidationError(
                record_id=None,
                field="record_id",
                reason="Missing record_id: field is None or undefined.",
            ))
        elif isinstance(record_id, str):
            if not record_id.strip():
                errors.append(ValidationError(
                    record_id=record_id,
                    field="record_id",
                    reason="Empty record_id: string must not be whitespace only.",
                ))
        elif not isinstance(record_id, int):
            errors.append(ValidationError(
                record_id=record_id,
                field="record_id",
                reason=f"Invalid record_id type: expected str or int, got {type(record_id).__name__}.",
            ))

        if record_id is not None and seen_ids is not None:
            if record_id in seen_ids:
                errors.append(ValidationError(
                    record_id=record_id,
                    field="record_id",
                    reason=f"Duplicate record_id: '{record_id}' already encountered.",
                ))
            else:
                seen_ids.add(record_id)

        # 2. text
        text = record.get("text")
        if text is None:
            errors.append(ValidationError(
                record_id=record_id,
                field="text",
                reason="Missing text: field is None or undefined.",
            ))
        elif not isinstance(text, str):
            errors.append(ValidationError(
                record_id=record_id,
                field="text",
                reason=f"Invalid text type: expected str, got {type(text).__name__}.",
            ))
        elif not text.strip():
            errors.append(ValidationError(
                record_id=record_id,
                field="text",
                reason="Empty text: string must not be empty or whitespace only.",
            ))

        # 3. label
        label = record.get("label")
        if label is None or isinstance(label, bool):
            errors.append(ValidationError(
                record_id=record_id,
                field="label",
                reason=f"Malformed label: expected integer in {sorted(self.allowed_labels)}, got {label!r}.",
            ))
        elif not isinstance(label, int):
            try:
                converted = float(label)
                if converted.is_integer() and int(converted) in self.allowed_labels:
                    label = int(converted)
                else:
                    errors.append(ValidationError(
                        record_id=record_id,
                        field="label",
                        reason=f"Label out of range: {label} is not in {sorted(self.allowed_labels)}.",
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    record_id=record_id,
                    field="label",
                    reason=f"Malformed label: unable to parse '{label}' as integer.",
                ))
        elif label not in self.allowed_labels:
            errors.append(ValidationError(
                record_id=record_id,
                field="label",
                reason=f"Label out of range: {label} not in {sorted(self.allowed_labels)}.",
            ))

        return errors

    def validate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        seen_ids: Set[Any] = set()
        valid_recs: List[Dict[str, Any]] = []
        all_errors: List[ValidationError] = []

        for r in records:
            errs = self.validate_record(r, seen_ids=seen_ids)
            if errs:
                all_errors.extend(errs)
            else:
                valid_recs.append(r)

        return valid_recs, all_errors


# =====================================================================
# Rigorous Evaluation Metrics with Explicit Zero-Denominator Handling
# =====================================================================

def calculate_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Calculate true positive, false positive, true negative, and false negative counts."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths must match: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}.")
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "total": len(y_true)}


def calculate_precision(
    y_true: List[int],
    y_pred: List[int],
    zero_division: float = 0.0,
) -> float:
    """
    Calculate Precision = TP / (TP + FP) with explicit zero-denominator safety.
    When TP + FP == 0, returns zero_division (default 0.0).
    """
    cm = calculate_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FP"]
    if denominator == 0:
        return zero_division
    return cm["TP"] / denominator


def calculate_recall(
    y_true: List[int],
    y_pred: List[int],
    zero_division: float = 0.0,
) -> float:
    """
    Calculate Recall = TP / (TP + FN) with explicit zero-denominator safety.
    When TP + FN == 0, returns zero_division (default 0.0).
    """
    cm = calculate_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FN"]
    if denominator == 0:
        return zero_division
    return cm["TP"] / denominator


def calculate_f1_score(
    y_true: List[int],
    y_pred: List[int],
    zero_division: float = 0.0,
) -> float:
    """Calculate harmonic mean F1 = 2 * (P * R) / (P + R) with zero-division safety."""
    p = calculate_precision(y_true, y_pred, zero_division=zero_division)
    r = calculate_recall(y_true, y_pred, zero_division=zero_division)
    if p + r == 0:
        return zero_division
    return 2.0 * (p * r) / (p + r)


def calculate_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate Accuracy = (TP + TN) / total."""
    cm = calculate_confusion_matrix(y_true, y_pred)
    if cm["total"] == 0:
        return 0.0
    return (cm["TP"] + cm["TN"]) / cm["total"]


def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Error = (1/N) * sum(|y_true - y_pred|).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths must match: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}.")
    if not y_true:
        raise ValueError("Cannot calculate MAE on empty sequence.")
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def calculate_token_exact_match(y_true_texts: List[str], y_pred_texts: List[str]) -> float:
    """Calculate exact match ratio for generated texts."""
    if len(y_true_texts) != len(y_pred_texts):
        raise ValueError("Lengths must match.")
    if not y_true_texts:
        return 0.0
    matches = sum(1 for yt, yp in zip(y_true_texts, y_pred_texts) if yt.strip() == yp.strip())
    return matches / len(y_true_texts)


def load_dataset(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON must be a list of records.")
            return data

    elif path.suffix.lower() == ".csv":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "record_id": row["record_id"],
                    "text": row["text"],
                    "label": int(row["label"]),
                })
        return records

    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
