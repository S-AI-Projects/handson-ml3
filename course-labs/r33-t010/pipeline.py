"""
R33-T010: Saved-Pipeline Interface and Schema Validation
Chapter 02: End-to-End Machine Learning Project / Data Cleaning

Contract:
- Save and reload regression pipeline configuration and parameters using safe JSON serialization.
- Never use or load untrusted pickle files.
- Expose a strict prediction function with schema validation (rejecting missing, extra, or non-finite features).
- Before/after predictions agree within explicit tolerance (1e-9).
- Schema/version mismatches fail clearly with explicit exceptions.
- Zero training data and zero credentials required at inference time.
- Deterministic formula fixture: 6 rows obeying target = 2*x1 + 1 (r1..r4 train, r5 val, r6 test).
- Offline, CPU-only, standard-library implementation with zero external dependencies.
- Labelled strictly as pipeline verification on synthetic data, NOT real-world performance evidence.
"""

from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union


SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}
EXPECTED_FEATURES = ["x1", "x2"]


class SchemaValidationError(Exception):
    """Raised when inference input data violates the feature schema."""
    pass


class VersionMismatchError(Exception):
    """Raised when an artifact schema_version is unsupported or incompatible."""
    pass


class ModelCorruptionError(Exception):
    """Raised when a saved model artifact is corrupted, incomplete, or malformed."""
    pass


@dataclass
class DatasetRecord:
    record_id: str
    x1: float
    x2: float
    target: float
    split: str

    def to_features(self) -> List[float]:
        return [self.x1, self.x2]

    def to_feature_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "x2": self.x2}


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


class SavedLinearPipeline:
    """
    Safely serializable linear regression pipeline.
    Serializes to clean, human-readable JSON avoiding pickle vulnerabilities.
    Validates feature schema and types at inference time.
    """

    def __init__(
        self,
        schema_version: str = "1.0.0",
        feature_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        intercept: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.schema_version = schema_version
        self.feature_names = feature_names if feature_names is not None else list(EXPECTED_FEATURES)
        self.weights = list(weights) if weights is not None else []
        self.intercept = intercept
        self.metadata = metadata if metadata is not None else {}
        self.is_fitted = len(self.weights) > 0

    def fit(self, records: List[DatasetRecord]) -> None:
        """
        Fit 2D linear regression on training records using closed-form normal equations.
        Solves w = (X^T X)^{-1} X^T y.
        """
        N = len(records)
        if N < 2:
            raise ValueError("Fitting requires at least 2 records.")

        X = [r.to_features() for r in records]
        y = [r.target for r in records]

        # Compute 2x2 X^T X
        XTX = [
            [sum(X[i][j] * X[i][k] for i in range(N)) for k in range(2)]
            for j in range(2)
        ]
        det = XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0]
        if abs(det) < 1e-12:
            raise ValueError("Singular matrix in linear fit.")

        inv = [
            [XTX[1][1] / det, -XTX[0][1] / det],
            [-XTX[0][1] / det, XTX[0][0] / det],
        ]

        XTy = [sum(X[i][j] * y[i] for i in range(N)) for j in range(2)]
        w = [sum(inv[j][k] * XTy[k] for k in range(2)) for j in range(2)]

        self.weights = [w[0], w[1]]
        self.intercept = 0.0
        self.metadata = {
            "num_training_samples": N,
            "training_split_ids": [r.record_id for r in records],
            "algorithm": "LinearRegression",
            "formula": "target = 2 * x1 + 1",
            "is_synthetic": True,
        }
        self.is_fitted = True

    def validate_single_input(self, row: Union[Dict[str, Any], List[float]]) -> List[float]:
        """
        Validate feature presence, count, types, and numerical finiteness.
        Returns ordered list of feature floats.
        """
        if isinstance(row, dict):
            # Check for missing features
            missing = [f for f in self.feature_names if f not in row]
            if missing:
                raise SchemaValidationError(
                    f"Missing required feature(s): {missing}. Expected features: {self.feature_names}"
                )
            # Check for extra unknown features
            extra = [f for f in row.keys() if f not in self.feature_names]
            if extra:
                raise SchemaValidationError(
                    f"Unknown feature(s) detected: {extra}. Allowed features: {self.feature_names}"
                )

            extracted = []
            for f in self.feature_names:
                val = row[f]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise SchemaValidationError(
                        f"Invalid type for feature '{f}': expected float or int, got {type(val).__name__}"
                    )
                if math.isnan(val) or math.isinf(val):
                    raise SchemaValidationError(
                        f"Non-finite value for feature '{f}': {val}"
                    )
                extracted.append(float(val))
            return extracted

        elif isinstance(row, (list, tuple)):
            if len(row) != len(self.feature_names):
                raise SchemaValidationError(
                    f"Incompatible feature count: expected {len(self.feature_names)} features, got {len(row)}"
                )
            extracted = []
            for idx, val in enumerate(row):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise SchemaValidationError(
                        f"Invalid type at index {idx}: expected float or int, got {type(val).__name__}"
                    )
                if math.isnan(val) or math.isinf(val):
                    raise SchemaValidationError(
                        f"Non-finite value at index {idx}: {val}"
                    )
                extracted.append(float(val))
            return extracted
        else:
            raise SchemaValidationError(
                f"Unsupported input row type: expected dict or list, got {type(row).__name__}"
            )

    def predict(
        self, inputs: Union[List[Dict[str, Any]], List[List[float]]]
    ) -> List[float]:
        """
        Inference function with strict schema validation.
        Requires NO training data or credentials.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict.")

        predictions = []
        for row in inputs:
            feats = self.validate_single_input(row)
            y_hat = self.intercept + sum(feats[j] * self.weights[j] for j in range(len(self.weights)))
            predictions.append(y_hat)
        return predictions

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline parameters to a serializable dictionary."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted pipeline.")
        return {
            "format": "safe_json_pipeline",
            "schema_version": self.schema_version,
            "feature_schema": {
                "feature_names": self.feature_names,
                "feature_types": {f: "float" for f in self.feature_names},
            },
            "parameters": {
                "weights": self.weights,
                "intercept": self.intercept,
            },
            "metadata": self.metadata,
        }

    def save(self, filepath: str) -> None:
        """Save pipeline to JSON artifact file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SavedLinearPipeline:
        """Construct pipeline from dictionary with schema and version checks."""
        if not isinstance(data, dict):
            raise ModelCorruptionError("Artifact root must be a JSON dictionary.")

        if data.get("format") != "safe_json_pipeline":
            raise ModelCorruptionError("Invalid or unrecognized model artifact format.")

        version = data.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise VersionMismatchError(
                f"Unsupported schema_version '{version}'. Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        schema_info = data.get("feature_schema", {})
        feature_names = schema_info.get("feature_names")
        if not feature_names or not isinstance(feature_names, list):
            raise ModelCorruptionError("Missing or invalid feature_schema in artifact.")

        params = data.get("parameters", {})
        weights = params.get("weights")
        if weights is None or not isinstance(weights, list) or len(weights) != len(feature_names):
            raise ModelCorruptionError("Missing or mismatched weights in artifact parameters.")

        intercept = float(params.get("intercept", 0.0))
        metadata = data.get("metadata", {})

        pipeline = cls(
            schema_version=version,
            feature_names=feature_names,
            weights=[float(w) for w in weights],
            intercept=intercept,
            metadata=metadata,
        )
        return pipeline

    @classmethod
    def load(cls, filepath: str) -> SavedLinearPipeline:
        """
        Safely load pipeline from JSON artifact file.
        Requires zero training data and zero external network credentials.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelCorruptionError(f"Corrupted or invalid JSON artifact: {e}")

        return cls.from_dict(data)
