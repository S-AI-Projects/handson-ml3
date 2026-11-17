"""
R33-T007: Reproducible Configuration and Parameter Schema for Regression Fixture
Chapter 02: End-to-End Machine Learning Project / Looking for Correlations

Contract:
- Serializable configuration specifying seed, schema_version, feature_ordering,
  split_ids, and algorithm_settings.
- Strict rejection of unknown fields, contradictory split IDs, and incompatible input shapes.
- Deterministic formula fixture: 6 rows obeying target = 2 * x1 + 1 (r1..r4 train, r5 val, r6 test).
- Reloaded configuration must deterministically reconstruct exact splits and predictions.
- Offline, CPU-only, standard-library implementation with zero external dependencies.
- Labelled strictly as pipeline verification on synthetic data, NOT real-world performance evidence.
"""

from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}
ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "seed",
    "feature_ordering",
    "target_column",
    "split_ids",
    "algorithm_settings",
}
ALLOWED_ALGORITHM_KEYS = {
    "algorithm",
    "fit_intercept",
    "regularization_alpha",
    "max_iter",
    "tolerance",
}
ALLOWED_ALGORITHMS = {"LinearRegression", "Ridge"}
VALID_FEATURES = {"x1", "x2"}


class ConfigValidationError(Exception):
    """Raised when configuration contains unknown fields or contradictory settings."""
    pass


class SchemaError(Exception):
    """Raised when input records violate the feature schema or shape."""
    pass


@dataclass
class DatasetRecord:
    record_id: str
    x1: float
    x2: float
    target: float
    split: str

    def get_features(self, feature_ordering: List[str]) -> List[float]:
        feats = []
        for f in feature_ordering:
            if f == "x1":
                feats.append(self.x1)
            elif f == "x2":
                feats.append(self.x2)
            else:
                raise SchemaError(f"Unknown feature requested: {f}")
        return feats


@dataclass
class AlgorithmSettings:
    algorithm: str = "LinearRegression"
    fit_intercept: bool = True
    regularization_alpha: float = 0.0
    max_iter: int = 100
    tolerance: float = 1e-5

    def validate(self) -> None:
        if self.algorithm not in ALLOWED_ALGORITHMS:
            raise ConfigValidationError(
                f"Unsupported algorithm '{self.algorithm}'. Allowed: {sorted(ALLOWED_ALGORITHMS)}"
            )
        if self.regularization_alpha < 0.0:
            raise ConfigValidationError(
                f"Contradictory setting: regularization_alpha cannot be negative ({self.regularization_alpha})"
            )
        if self.max_iter <= 0:
            raise ConfigValidationError(
                f"Contradictory setting: max_iter must be positive ({self.max_iter})"
            )
        if self.tolerance <= 0.0:
            raise ConfigValidationError(
                f"Contradictory setting: tolerance must be positive ({self.tolerance})"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlgorithmSettings:
        unknown_keys = set(data.keys()) - ALLOWED_ALGORITHM_KEYS
        if unknown_keys:
            raise ConfigValidationError(
                f"Unknown algorithm setting(s) detected: {sorted(unknown_keys)}"
            )
        settings = cls(
            algorithm=data.get("algorithm", "LinearRegression"),
            fit_intercept=data.get("fit_intercept", True),
            regularization_alpha=float(data.get("regularization_alpha", 0.0)),
            max_iter=int(data.get("max_iter", 100)),
            tolerance=float(data.get("tolerance", 1e-5)),
        )
        settings.validate()
        return settings


@dataclass
class ExperimentConfig:
    schema_version: str
    seed: int
    feature_ordering: List[str]
    target_column: str
    split_ids: Dict[str, List[str]]
    algorithm_settings: AlgorithmSettings

    def validate(self, available_record_ids: Optional[Set[str]] = None) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ConfigValidationError(
                f"Unsupported schema_version '{self.schema_version}'. Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        if not self.feature_ordering:
            raise ConfigValidationError("Feature ordering cannot be empty.")
        for feat in self.feature_ordering:
            if feat not in VALID_FEATURES:
                raise ConfigValidationError(
                    f"Unknown feature in ordering: '{feat}'. Allowed: {sorted(VALID_FEATURES)}"
                )
        if len(self.feature_ordering) != len(set(self.feature_ordering)):
            raise ConfigValidationError("Duplicate features in feature_ordering.")

        if self.target_column != "target":
            raise ConfigValidationError(
                f"Invalid target column '{self.target_column}'. Expected 'target'."
            )

        # Validate splits
        required_splits = {"train", "validation", "test"}
        if set(self.split_ids.keys()) != required_splits:
            raise ConfigValidationError(
                f"Split keys mismatch. Expected {sorted(required_splits)}, got {sorted(self.split_ids.keys())}"
            )

        for split_name in required_splits:
            ids = self.split_ids[split_name]
            if not ids:
                raise ConfigValidationError(f"Split '{split_name}' cannot be empty.")

        # Check for contradictory overlapping IDs across splits
        train_set = set(self.split_ids["train"])
        val_set = set(self.split_ids["validation"])
        test_set = set(self.split_ids["test"])

        if train_set & val_set:
            raise ConfigValidationError(
                f"Contradictory split IDs: Overlap between train and validation: {sorted(train_set & val_set)}"
            )
        if train_set & test_set:
            raise ConfigValidationError(
                f"Contradictory split IDs: Overlap between train and test: {sorted(train_set & test_set)}"
            )
        if val_set & test_set:
            raise ConfigValidationError(
                f"Contradictory split IDs: Overlap between validation and test: {sorted(val_set & test_set)}"
            )

        if available_record_ids is not None:
            all_split_ids = train_set | val_set | test_set
            unknown_ids = all_split_ids - available_record_ids
            if unknown_ids:
                raise ConfigValidationError(
                    f"Contradictory configuration: Split IDs not found in dataset: {sorted(unknown_ids)}"
                )

        self.algorithm_settings.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "feature_ordering": list(self.feature_ordering),
            "target_column": self.target_column,
            "split_ids": {k: list(v) for k, v in self.split_ids.items()},
            "algorithm_settings": asdict(self.algorithm_settings),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], available_record_ids: Optional[Set[str]] = None
    ) -> ExperimentConfig:
        if not isinstance(data, dict):
            raise ConfigValidationError("Configuration root must be a JSON dictionary.")

        unknown_keys = set(data.keys()) - ALLOWED_CONFIG_KEYS
        if unknown_keys:
            raise ConfigValidationError(
                f"Unknown configuration field(s) detected: {sorted(unknown_keys)}"
            )

        missing_keys = ALLOWED_CONFIG_KEYS - set(data.keys())
        if missing_keys:
            raise ConfigValidationError(
                f"Missing required configuration field(s): {sorted(missing_keys)}"
            )

        algo_data = data.get("algorithm_settings", {})
        if not isinstance(algo_data, dict):
            raise ConfigValidationError("algorithm_settings must be a dictionary.")

        settings = AlgorithmSettings.from_dict(algo_data)

        config = cls(
            schema_version=str(data["schema_version"]),
            seed=int(data["seed"]),
            feature_ordering=list(data["feature_ordering"]),
            target_column=str(data["target_column"]),
            split_ids={k: list(v) for k, v in data["split_ids"].items()},
            algorithm_settings=settings,
        )
        config.validate(available_record_ids=available_record_ids)
        return config

    @classmethod
    def from_json(
        cls, json_str: str, available_record_ids: Optional[Set[str]] = None
    ) -> ExperimentConfig:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON configuration: {e}")
        return cls.from_dict(data, available_record_ids=available_record_ids)


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

    # Assert deterministic formula
    for r in records:
        expected = 2.0 * r.x1 + 1.0
        if abs(r.target - expected) > 1e-7:
            raise SchemaError(
                f"Record {r.record_id} violates formula target = 2*x1 + 1 (got {r.target}, expected {expected})"
            )

    return records


class DeterministicRegressor:
    """
    Closed-form Linear/Ridge Regressor using 2x2 normal equations.
    Solves (X^T X + alpha * I) w = X^T y.
    """

    def __init__(self, regularization_alpha: float = 0.0):
        self.regularization_alpha = regularization_alpha
        self.weights: List[float] = []
        self.feature_names: List[str] = []

    def fit(self, X: List[List[float]], y: List[float], feature_names: List[str]) -> None:
        N = len(X)
        D = len(X[0])
        self.feature_names = list(feature_names)

        if D == 1:
            # 1D with alpha: (sum(x^2) + alpha) * w = sum(x*y)
            denom = sum(X[i][0] ** 2 for i in range(N)) + self.regularization_alpha
            if abs(denom) < 1e-12:
                raise ValueError("Singular matrix in 1D regression.")
            num = sum(X[i][0] * y[i] for i in range(N))
            self.weights = [num / denom]
            return

        if D == 2:
            # Compute 2x2 X^T X
            XTX = [
                [sum(X[i][j] * X[i][k] for i in range(N)) for k in range(2)]
                for j in range(2)
            ]
            # Add Ridge regularization
            XTX[0][0] += self.regularization_alpha
            XTX[1][1] += self.regularization_alpha

            det = XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0]
            if abs(det) < 1e-12:
                raise ValueError("Singular feature matrix: columns are collinear.")

            inv = [
                [XTX[1][1] / det, -XTX[0][1] / det],
                [-XTX[1][0] / det, XTX[0][0] / det],
            ]

            XTy = [sum(X[i][j] * y[i] for i in range(N)) for j in range(2)]
            self.weights = [sum(inv[j][k] * XTy[k] for k in range(2)) for j in range(2)]
            return

        raise NotImplementedError(f"Regressor supports up to 2 features, got {D}.")

    def predict(self, X: List[List[float]]) -> List[float]:
        if not self.weights:
            raise RuntimeError("Model is not fitted.")
        D = len(self.weights)
        preds = []
        for row in X:
            if len(row) != D:
                raise SchemaError(
                    f"Incompatible input shape: Expected {D} features, got {len(row)}."
                )
            y_hat = sum(row[j] * self.weights[j] for j in range(D))
            preds.append(y_hat)
        return preds


@dataclass
class PipelineResult:
    config: ExperimentConfig
    train_record_ids: List[str]
    val_record_ids: List[str]
    test_record_ids: List[str]
    train_predictions: Dict[str, float]
    val_predictions: Dict[str, float]
    test_predictions: Dict[str, float]
    weights: Dict[str, float]
    train_mse: float
    val_mse: float
    test_mse: float
    is_synthetic: bool = True
    cpu_only: bool = True


def run_pipeline(
    records: List[DatasetRecord], config: ExperimentConfig
) -> PipelineResult:
    """Execute pipeline using the serializable configuration."""
    record_map = {r.record_id: r for r in records}
    available_ids = set(record_map.keys())

    # Re-validate against dataset records
    config.validate(available_record_ids=available_ids)

    # Partition data
    train_records = [record_map[rid] for rid in config.split_ids["train"]]
    val_records = [record_map[rid] for rid in config.split_ids["validation"]]
    test_records = [record_map[rid] for rid in config.split_ids["test"]]

    # Build feature matrices
    X_train = [r.get_features(config.feature_ordering) for r in train_records]
    y_train = [r.target for r in train_records]

    X_val = [r.get_features(config.feature_ordering) for r in val_records]
    y_val = [r.target for r in val_records]

    X_test = [r.get_features(config.feature_ordering) for r in test_records]
    y_test = [r.target for r in test_records]

    # Fit model
    regressor = DeterministicRegressor(
        regularization_alpha=config.algorithm_settings.regularization_alpha
    )
    regressor.fit(X_train, y_train, feature_names=config.feature_ordering)

    # Predict
    preds_train = regressor.predict(X_train)
    preds_val = regressor.predict(X_val)
    preds_test = regressor.predict(X_test)

    train_pred_dict = {
        r.record_id: preds_train[i] for i, r in enumerate(train_records)
    }
    val_pred_dict = {
        r.record_id: preds_val[i] for i, r in enumerate(val_records)
    }
    test_pred_dict = {
        r.record_id: preds_test[i] for i, r in enumerate(test_records)
    }

    # Calculate MSE
    train_mse = sum((preds_train[i] - y_train[i]) ** 2 for i in range(len(y_train))) / len(y_train)
    val_mse = sum((preds_val[i] - y_val[i]) ** 2 for i in range(len(y_val))) / len(y_val)
    test_mse = sum((preds_test[i] - y_test[i]) ** 2 for i in range(len(y_test))) / len(y_test)

    weights_dict = {
        feat: regressor.weights[i] for i, feat in enumerate(config.feature_ordering)
    }

    return PipelineResult(
        config=config,
        train_record_ids=config.split_ids["train"],
        val_record_ids=config.split_ids["validation"],
        test_record_ids=config.split_ids["test"],
        train_predictions=train_pred_dict,
        val_predictions=val_pred_dict,
        test_predictions=test_pred_dict,
        weights=weights_dict,
        train_mse=train_mse,
        val_mse=val_mse,
        test_mse=test_mse,
        is_synthetic=True,
        cpu_only=True,
    )
