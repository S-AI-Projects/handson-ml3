"""
Course Lab R33-T001: Data Contract Schema Validator & Stratified Split.

Scope: 03 classification / MNIST input validation component.
Note: Verification fixture rows are synthetic and intended strictly for pipeline
schema and contract testing. They are NOT observations from real MNIST and should
not be used for real-world or clinical performance claims.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
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
    Validates classification input records against the R33-T001 data contract:
      - record_id: Unique identifier (non-empty string or integer).
      - group_id: Group identifier (non-empty string or integer).
      - x1: Numeric scalar (int or float, non-NaN, finite).
      - x2: Numeric scalar (int or float, non-NaN, finite).
      - label: Binary class label in {0, 1}.
    """

    REQUIRED_FIELDS = ("record_id", "group_id", "x1", "x2", "label")

    def __init__(self) -> None:
        self.seen_record_ids: set = set()

    def reset(self) -> None:
        self.seen_record_ids.clear()

    def validate_row(self, row: Dict[str, Any], fallback_id: str) -> List[ValidationError]:
        """Validates a single record dictionary and returns a list of any errors found."""
        errors: List[ValidationError] = []

        # 1. Validate record_id
        raw_rec_id = row.get("record_id")
        effective_id = raw_rec_id if raw_rec_id is not None and str(raw_rec_id).strip() != "" else fallback_id

        if raw_rec_id is None or str(raw_rec_id).strip() == "":
            errors.append(ValidationError(
                record_id=fallback_id,
                field="record_id",
                reason="Missing or empty record_id"
            ))
        elif raw_rec_id in self.seen_record_ids:
            errors.append(ValidationError(
                record_id=raw_rec_id,
                field="record_id",
                reason=f"Duplicate record_id: {raw_rec_id}"
            ))
        else:
            self.seen_record_ids.add(raw_rec_id)

        # 2. Validate group_id
        raw_grp_id = row.get("group_id")
        if raw_grp_id is None or str(raw_grp_id).strip() == "":
            errors.append(ValidationError(
                record_id=effective_id,
                field="group_id",
                reason="Missing or empty group_id"
            ))

        # 3. Validate numeric features: x1, x2
        for feature in ("x1", "x2"):
            if feature not in row or row[feature] is None:
                errors.append(ValidationError(
                    record_id=effective_id,
                    field=feature,
                    reason=f"Missing required numeric feature '{feature}'"
                ))
                continue

            val = row[feature]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errors.append(ValidationError(
                    record_id=effective_id,
                    field=feature,
                    reason=f"Malformed numeric data: '{feature}' must be int or float, got {type(val).__name__} ({val!r})"
                ))
            elif math.isnan(val) or math.isinf(val):
                errors.append(ValidationError(
                    record_id=effective_id,
                    field=feature,
                    reason=f"Malformed numeric data: '{feature}' must be finite and non-NaN, got {val!r}"
                ))

        # 4. Validate binary label
        if "label" not in row or row["label"] is None:
            errors.append(ValidationError(
                record_id=effective_id,
                field="label",
                reason="Missing required field 'label'"
            ))
        else:
            lbl = row["label"]
            if isinstance(lbl, bool) or not isinstance(lbl, int) or lbl not in (0, 1):
                errors.append(ValidationError(
                    record_id=effective_id,
                    field="label",
                    reason=f"Invalid label: expected integer in {{0, 1}}, got {type(lbl).__name__} ({lbl!r})"
                ))

        return errors

    def validate_records(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        """
        Validates all records in the dataset.
        Returns:
            Tuple of (valid_records, list_of_all_validation_errors).
            No invalid record is silently dropped; all failures are reported.
        """
        self.reset()
        valid_records: List[Dict[str, Any]] = []
        all_errors: List[ValidationError] = []

        for idx, row in enumerate(records):
            fallback_id = f"row_{idx}"
            row_errors = self.validate_row(row, fallback_id)
            if row_errors:
                all_errors.extend(row_errors)
            else:
                valid_records.append(row)

        return valid_records, all_errors


def load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Loads records from a JSON file, CSV file, or returns in-memory list of dicts."""
    if isinstance(source, list):
        return source

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Dataset source path not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a top-level list of objects.")
            return data

    if p.suffix.lower() == ".csv":
        records = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row: Dict[str, Any] = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    v_clean = v.strip()
                    if k_clean in ("x1", "x2"):
                        try:
                            parsed_row[k_clean] = float(v_clean) if "." in v_clean else int(v_clean)
                        except ValueError:
                            parsed_row[k_clean] = v_clean
                    elif k_clean == "label":
                        try:
                            parsed_row[k_clean] = int(v_clean)
                        except ValueError:
                            parsed_row[k_clean] = v_clean
                    else:
                        parsed_row[k_clean] = v_clean
                records.append(parsed_row)
        return records

    raise ValueError(f"Unsupported file format: {p.suffix}. Expected .json or .csv")


def fixed_stratified_split(
    records: List[Dict[str, Any]], train_ratio: float = 0.5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes a deterministic, fixed stratified split by label without external dependencies.
    Ensures balanced label proportions across train and test partitions.
    """
    by_label: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        lbl = r["label"]
        by_label.setdefault(lbl, []).append(r)

    train_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for lbl in sorted(by_label.keys()):
        group = by_label[lbl]
        n_train = max(1, round(len(group) * train_ratio))
        train_set.extend(group[:n_train])
        test_set.extend(group[n_train:])

    return train_set, test_set
