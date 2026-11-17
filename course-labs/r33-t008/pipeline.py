"""
R33-T008: Error-Analysis Report for Linear Regression Fixture
Chapter 04: Training Linear Models / Learning Curves

Contract:
- Create a structured error-analysis report for the regression fixture:
  stable record ID, expected value, predicted value, and reason category.
- Add one deliberately wrong prediction so the report is testable.
- Done when:
  1. The injected wrong prediction appears in the report with the correct record ID.
  2. Correct predictions do not appear in the error report detailed rows.
  3. Aggregate error counts match the detailed rows exactly.
  4. Six synthetic rows with x1=[0..5], x2=1, target=2*x1+1 (r1..r4 train, r5 val, r6 test).
- Offline, CPU-only, standard-library implementation with zero external dependencies.
- Labelled strictly as pipeline verification on synthetic data, NOT real-world performance evidence.
"""

from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


REASON_CATEGORIES = {
    "injected_fault",
    "sensor_saturation_artifact",
    "extrapolation_uncertainty",
    "data_contamination",
    "unspecified_residual",
}


@dataclass
class DatasetRecord:
    record_id: str
    x1: float
    x2: float
    target: float
    split: str

    def to_features(self) -> List[float]:
        return [self.x1, self.x2]


@dataclass
class ErrorRecord:
    record_id: str
    split: str
    x1: float
    x2: float
    expected_value: float
    predicted_value: float
    residual: float
    absolute_error: float
    relative_error: float
    reason_category: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorAnalysisReport:
    total_evaluated: int
    total_correct: int
    total_errors: int
    error_threshold: float
    mean_absolute_error: float
    max_absolute_error: float
    error_records: List[ErrorRecord]
    aggregate_counts_by_reason: Dict[str, int]
    is_synthetic: bool = True
    cpu_only: bool = True

    def validate_integrity(self) -> None:
        """Assert report mathematical consistency."""
        if self.total_evaluated != self.total_correct + self.total_errors:
            raise AssertionError(
                f"Evaluation count mismatch: evaluated ({self.total_evaluated}) != "
                f"correct ({self.total_correct}) + errors ({self.total_errors})"
            )
        if self.total_errors != len(self.error_records):
            raise AssertionError(
                f"Error count mismatch: total_errors ({self.total_errors}) != "
                f"len(error_records) ({len(self.error_records)})"
            )
        sum_category_counts = sum(self.aggregate_counts_by_reason.values())
        if sum_category_counts != self.total_errors:
            raise AssertionError(
                f"Aggregate category sum ({sum_category_counts}) != total_errors ({self.total_errors})"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate_integrity()
        return {
            "total_evaluated": self.total_evaluated,
            "total_correct": self.total_correct,
            "total_errors": self.total_errors,
            "error_threshold": self.error_threshold,
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "aggregate_counts_by_reason": self.aggregate_counts_by_reason,
            "error_records": [rec.to_dict() for rec in self.error_records],
            "is_synthetic": self.is_synthetic,
            "cpu_only": self.cpu_only,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def load_dataset(filepath: Optional[str] = None, format: str = "json") -> List[DatasetRecord]:
    """Load the 6-row synthetic regression fixture from JSON or CSV."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "fixture.json" if format.lower() == "json" else "fixture.csv"
        filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fixture file not found: {filepath}")

    records: List[DatasetRecord] = []

    if format.lower() == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["records"]:
            record = DatasetRecord(
                record_id=item["record_id"],
                x1=float(item["x1"]),
                x2=float(item["x2"]),
                target=float(item["target"]),
                split=item.get("split", "train"),
            )
            records.append(record)

    elif format.lower() == "csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = DatasetRecord(
                    record_id=row["record_id"],
                    x1=float(row["x1"]),
                    x2=float(row["x2"]),
                    target=float(row["target"]),
                    split=row.get("split", "train"),
                )
                records.append(record)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Validate deterministic formula: target = 2 * x1 + 1
    for r in records:
        expected = 2.0 * r.x1 + 1.0
        if abs(r.target - expected) > 1e-7:
            raise ValueError(
                f"Record {r.record_id} violates formula target = 2*x1 + 1: "
                f"target={r.target}, expected={expected}"
            )

    return records


class DeterministicLinearModel:
    """
    Closed-form 2D linear regression solver:
    Computes w = (X^T X)^{-1} X^T y.
    Learns w_1 = 2.0, w_2 = 1.0 on training records r1..r4.
    """

    def __init__(self):
        self.w1: float = 0.0
        self.w2: float = 0.0
        self.is_fitted: bool = False

    def fit(self, records: List[DatasetRecord]) -> None:
        N = len(records)
        if N < 2:
            raise ValueError("Fitting requires at least 2 training records.")

        X = [r.to_features() for r in records]
        y = [r.target for r in records]

        # Compute 2x2 X^T X
        XTX = [
            [sum(X[i][j] * X[i][k] for i in range(N)) for k in range(2)]
            for j in range(2)
        ]
        det = XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0]
        if abs(det) < 1e-12:
            raise ValueError("Singular matrix: feature columns are collinear.")

        inv = [
            [XTX[1][1] / det, -XTX[0][1] / det],
            [-XTX[0][1] / det, XTX[0][0] / det],
        ]

        XTy = [sum(X[i][j] * y[i] for i in range(N)) for j in range(2)]
        w = [sum(inv[j][k] * XTy[k] for k in range(2)) for j in range(2)]

        self.w1 = w[0]
        self.w2 = w[1]
        self.is_fitted = True

    def predict_record(self, record: DatasetRecord) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.w1 * record.x1 + self.w2 * record.x2

    def predict_all(self, records: List[DatasetRecord]) -> Dict[str, float]:
        return {r.record_id: self.predict_record(r) for r in records}


def analyze_predictions(
    records: List[DatasetRecord],
    predictions: Dict[str, float],
    error_threshold: float = 0.01,
    reason_map: Optional[Dict[str, str]] = None,
) -> ErrorAnalysisReport:
    """
    Generate an ErrorAnalysisReport comparing predicted values against expected values.
    Records with absolute error <= error_threshold are considered correct and OMITTED from error details.
    Records with absolute error > error_threshold are classified with a reason category.
    """
    if reason_map is None:
        reason_map = {}

    total_evaluated = len(records)
    total_correct = 0
    error_records: List[ErrorRecord] = []
    category_counts: Dict[str, int] = {}
    abs_errors: List[float] = []

    for r in records:
        if r.record_id not in predictions:
            raise KeyError(f"Missing prediction for record ID: {r.record_id}")

        expected = r.target
        pred = predictions[r.record_id]
        residual = pred - expected
        abs_err = abs(residual)
        abs_errors.append(abs_err)

        if abs_err <= error_threshold:
            total_correct += 1
        else:
            rel_err = abs_err / abs(expected) if abs(expected) > 1e-9 else float("inf")
            reason = reason_map.get(r.record_id, "unspecified_residual")

            rec = ErrorRecord(
                record_id=r.record_id,
                split=r.split,
                x1=r.x1,
                x2=r.x2,
                expected_value=expected,
                predicted_value=pred,
                residual=residual,
                absolute_error=abs_err,
                relative_error=rel_err,
                reason_category=reason,
            )
            error_records.append(rec)
            category_counts[reason] = category_counts.get(reason, 0) + 1

    total_errors = len(error_records)
    mean_abs_err = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    max_abs_err = max(abs_errors) if abs_errors else 0.0

    report = ErrorAnalysisReport(
        total_evaluated=total_evaluated,
        total_correct=total_correct,
        total_errors=total_errors,
        error_threshold=error_threshold,
        mean_absolute_error=mean_abs_err,
        max_absolute_error=max_abs_err,
        error_records=error_records,
        aggregate_counts_by_reason=category_counts,
        is_synthetic=True,
        cpu_only=True,
    )
    report.validate_integrity()
    return report
