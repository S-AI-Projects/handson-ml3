"""
Course Lab R33-T005: Preprocessing Contract & Parameter Isolation.

Scope: 04 training linear models / Stochastic Gradient Descent.
Note: Verification fixture rows are synthetic (6 rows with x1=[0..5], x2=1, target=2*x1+1).
They are strictly intended for preprocessing contract validation and parameter isolation.
They are NOT observations from any real dataset and must not be used for performance claims.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import csv
import json
import math
from pathlib import Path


class ContaminationError(Exception):
    """Raised when held-out data or deliberate contamination is introduced to fit()."""
    pass


class NotFittedError(Exception):
    """Raised when transform is called before fit."""
    pass


class SchemaError(Exception):
    """Raised when record data fails the schema contract."""
    pass


@dataclass
class PreprocessingState:
    """Stores fitted preprocessing statistics computed strictly on training data."""
    feature_order: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    impute_values: Dict[str, float]
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "means": self.means,
            "stds": self.stds,
            "impute_values": self.impute_values,
            "sample_count": self.sample_count,
        }


class ExplicitPreprocessor:
    """
    Fits missing-value and scale parameters strictly on training rows.
    Guarantees:
      - Fitting on held-out inputs is impossible via transform().
      - Rejects deliberate contamination fixtures.
      - Retains exact numeric float types and deterministic feature ordering.
    """

    ALLOWED_TRAIN_SPLIT_LABELS: Set[str] = {"train", "training"}
    PROHIBITED_SPLIT_LABELS: Set[str] = {"val", "validation", "test", "held_out"}

    def __init__(self, feature_order: Sequence[str] = ("x1", "x2")) -> None:
        self.feature_order: List[str] = list(feature_order)
        self.state_: Optional[PreprocessingState] = None

    @property
    def is_fitted(self) -> bool:
        return self.state_ is not None

    def fit(
        self,
        train_records: List[Dict[str, Any]],
        prohibited_record_ids: Optional[Set[str]] = None,
    ) -> "ExplicitPreprocessor":
        """
        Fits scale and imputation statistics strictly on training data.
        Raises ContaminationError if held-out records or deliberate contamination is detected.
        """
        if not train_records:
            raise SchemaError("Training dataset cannot be empty.")

        if prohibited_record_ids is None:
            prohibited_record_ids = {"r5", "r6"}  # Default validation/test IDs

        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        impute_values: Dict[str, float] = {}

        # 1. Contamination & isolation checks
        for idx, r in enumerate(train_records):
            rec_id = str(r.get("record_id", "")).strip()

            # Check if record is flagged with a prohibited split label
            split_label = str(r.get("split", "")).strip().lower()
            if split_label in self.PROHIBITED_SPLIT_LABELS:
                raise ContaminationError(
                    f"Contamination error: Record '{rec_id}' (index {idx}) has prohibited split '{split_label}' in training input."
                )

            # Check if record matches known held-out IDs
            if rec_id in prohibited_record_ids:
                raise ContaminationError(
                    f"Contamination error: Held-out record ID '{rec_id}' was supplied to fit()."
                )

            # Check for deliberate contamination marker
            if r.get("_contaminated", False) or r.get("is_contaminated", False):
                raise ContaminationError(
                    f"Contamination error: Deliberate contamination flag detected in record '{rec_id}'."
                )

        # 2. Compute training column statistics for each feature in feature_order
        n = len(train_records)
        for feat in self.feature_order:
            valid_vals: List[float] = []
            for r in train_records:
                val = r.get(feat)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise SchemaError(f"Field '{feat}' contains invalid non-numeric value: {val!r}")
                if math.isinf(val):
                    raise ContaminationError(f"Contamination: Infinite value detected in feature '{feat}'")
                valid_vals.append(float(val))

            if not valid_vals:
                raise SchemaError(f"Feature '{feat}' has no valid numeric values to fit.")

            mean_val = sum(valid_vals) / len(valid_vals)
            # Population variance on training observations
            variance = sum((v - mean_val) ** 2 for v in valid_vals) / len(valid_vals)
            std_val = math.sqrt(variance)

            # If constant feature (variance == 0), set scale divisor to 1.0 to prevent div-by-zero
            if std_val == 0.0 or math.isclose(std_val, 0.0, abs_tol=1e-9):
                std_val = 1.0

            means[feat] = mean_val
            stds[feat] = std_val
            impute_values[feat] = mean_val

        self.state_ = PreprocessingState(
            feature_order=list(self.feature_order),
            means=means,
            stds=stds,
            impute_values=impute_values,
            sample_count=n,
        )
        return self

    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms records using strictly training-fitted parameters.
        Transform is read-only: it does NOT alter or re-estimate any fitted state.
        """
        if not self.is_fitted or self.state_ is None:
            raise NotFittedError("ExplicitPreprocessor must be fit on training data before transform.")

        transformed: List[Dict[str, Any]] = []

        for idx, r in enumerate(records):
            rec_id = r.get("record_id", f"row_{idx}")

            # Check for deliberate contamination
            if r.get("_contaminated", False) or r.get("is_contaminated", False):
                raise ContaminationError(
                    f"Contamination error in transform: Deliberate contamination flag in record '{rec_id}'."
                )

            row_out: Dict[str, Any] = {
                "record_id": rec_id,
            }
            if "split" in r:
                row_out["split"] = r["split"]
            if "target" in r:
                row_out["target"] = float(r["target"]) if isinstance(r["target"], (int, float)) else r["target"]

            # Process features strictly in feature_order
            feature_vector: List[float] = []
            for feat in self.state_.feature_order:
                val = r.get(feat)
                # Missing value imputation
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    imputed = self.state_.impute_values[feat]
                elif isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise SchemaError(f"Record '{rec_id}': '{feat}' must be numeric, got {type(val).__name__}")
                elif math.isinf(val):
                    raise ContaminationError(f"Record '{rec_id}': Infinite value detected in '{feat}'")
                else:
                    imputed = float(val)

                # Standard scaling: (x - mean) / std
                scaled = (imputed - self.state_.means[feat]) / self.state_.stds[feat]
                row_out[feat] = float(scaled)
                feature_vector.append(float(scaled))

            row_out["_feature_vector"] = feature_vector
            transformed.append(row_out)

        return transformed

    def fit_transform(self, train_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convenience method to fit on train records and transform them."""
        return self.fit(train_records).transform(train_records)


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
                    "x1": float(row["x1"].strip()) if "." in row["x1"] else int(row["x1"].strip()),
                    "x2": float(row["x2"].strip()) if "." in row["x2"] else int(row["x2"].strip()),
                    "target": float(row["target"].strip()) if "." in row["target"] else int(row["target"].strip()),
                    "split": row.get("split", "").strip(),
                })
        return records

    raise ValueError(f"Unsupported format: {p.suffix}")
