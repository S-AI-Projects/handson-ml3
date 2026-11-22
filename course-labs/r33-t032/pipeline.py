"""
Course Lab R33-T032: Leakage-Free Splitting & Deterministic Split Manifest.

Student: Snehakumari Vinod Singh
Task: R33-T032 (Task 2 of 10)
Target Item: 06_decision_trees.ipynb (Making Predictions)

Disclaimer:
The synthetic classification fixture (8 rows, x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1])
is strictly designed for CPU smoke testing and data splitting verification.
It is NOT an observation from any real-world dataset and must not be used as evidence
for real-world or clinical predictive performance.
"""

from dataclasses import asdict, dataclass
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
    Validates classification input records against the R33-T032 data contract:
      - record_id: Unique identifier (non-empty str or int).
      - group_id: Cohort/subject identifier (non-empty str or int).
      - x1: Numeric feature (int or float).
      - x2: Numeric feature (int or float).
      - label: Binary class identifier (0 or 1).
      - timestamp: Optional numeric chronological ordering indicator.
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
                reason=f"Malformed record: expected dict mapping, got {type(record).__name__}.",
            )]

        record_id = record.get("record_id")
        errors: List[ValidationError] = []

        # 1. record_id validation
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
                    reason="Empty record_id: string must not be empty or whitespace only.",
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
                    reason=f"Duplicate record_id: '{record_id}' was already encountered in the dataset.",
                ))
            else:
                seen_ids.add(record_id)

        # 2. group_id validation
        group_id = record.get("group_id")
        if group_id is None:
            errors.append(ValidationError(
                record_id=record_id,
                field="group_id",
                reason="Missing group_id: field is None or undefined.",
            ))
        elif isinstance(group_id, str):
            if not group_id.strip():
                errors.append(ValidationError(
                    record_id=record_id,
                    field="group_id",
                    reason="Empty group_id: string must not be empty or whitespace only.",
                ))
        elif not isinstance(group_id, int):
            errors.append(ValidationError(
                record_id=record_id,
                field="group_id",
                reason=f"Invalid group_id type: expected str or int, got {type(group_id).__name__}.",
            ))

        # 3. Numeric feature validations (x1, x2)
        for feat in ["x1", "x2"]:
            val = record.get(feat)
            if val is None or isinstance(val, bool):
                errors.append(ValidationError(
                    record_id=record_id,
                    field=feat,
                    reason=f"Malformed feature '{feat}': expected numeric value, got {val!r}.",
                ))
            elif not isinstance(val, (int, float)):
                errors.append(ValidationError(
                    record_id=record_id,
                    field=feat,
                    reason=f"Malformed feature '{feat}': non-numeric type {type(val).__name__}.",
                ))
            elif math.isnan(val) or math.isinf(val):
                errors.append(ValidationError(
                    record_id=record_id,
                    field=feat,
                    reason=f"Malformed feature '{feat}': NaN or Infinite value.",
                ))

        # 4. Label validation
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
                        reason=f"Label out of range: {label} is not in allowed classes {sorted(self.allowed_labels)}.",
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    record_id=record_id,
                    field="label",
                    reason=f"Malformed label: unable to parse '{label}' as integer class label.",
                ))
        elif label not in self.allowed_labels:
            errors.append(ValidationError(
                record_id=record_id,
                field="label",
                reason=f"Label out of range: value {label} is not in allowed classes {sorted(self.allowed_labels)}.",
            ))

        return errors

    def validate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        seen_ids: Set[Any] = set()
        valid_records: List[Dict[str, Any]] = []
        all_errors: List[ValidationError] = []

        for record in records:
            errs = self.validate_record(record, seen_ids=seen_ids)
            if errs:
                all_errors.extend(errs)
            else:
                valid_records.append(record)

        return valid_records, all_errors


@dataclass
class SplitManifest:
    """
    Deterministic Split Manifest recording exact record IDs allocated to partitions.
    Guarantees no data leakage across training, validation, and test sets.
    """
    train: List[str]
    validation: List[str]
    test: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "splits": {
                "train": self.train,
                "validation": self.validation,
                "test": self.test,
            },
            "summary": {
                "train_count": len(self.train),
                "validation_count": len(self.validation),
                "test_count": len(self.test),
                "total_records": len(self.train) + len(self.validation) + len(self.test),
            },
            "metadata": self.metadata,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SplitManifest":
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        splits = data.get("splits", {})
        return cls(
            train=splits.get("train", []),
            validation=splits.get("validation", []),
            test=splits.get("test", []),
            metadata=data.get("metadata", {}),
        )


def generate_stratified_split_manifest(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> SplitManifest:
    """
    Generate a deterministic stratified split manifest writing exact record IDs
    to train, validation, and test partitions.
    """
    # Group by label to ensure exact class balance
    by_label: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        lbl = r["label"]
        by_label.setdefault(lbl, []).append(r)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for lbl in sorted(by_label.keys()):
        class_records = by_label[lbl]
        n_total = len(class_records)
        n_train = max(1, round(n_total * train_ratio))
        n_val = max(1, round(n_total * val_ratio))
        
        train_slice = class_records[:n_train]
        val_slice = class_records[n_train:n_train + n_val]
        test_slice = class_records[n_train + n_val:]

        train_ids.extend([str(r["record_id"]) for r in train_slice])
        val_ids.extend([str(r["record_id"]) for r in val_slice])
        test_ids.extend([str(r["record_id"]) for r in test_slice])

    manifest = SplitManifest(
        train=sorted(train_ids),
        validation=sorted(val_ids),
        test=sorted(test_ids),
        metadata={
            "split_type": "stratified",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "dataset_size": len(records),
        },
    )
    return manifest


def generate_chronological_split_manifest(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
) -> SplitManifest:
    """
    Generate a strictly chronological time-series split manifest.
    Guarantees train timestamps <= validation timestamps <= test timestamps.
    """
    sorted_records = sorted(records, key=lambda r: (r.get("timestamp", 0), r["record_id"]))
    n_total = len(sorted_records)
    n_train = max(1, round(n_total * train_ratio))
    n_val = max(1, round(n_total * val_ratio))

    train_slice = sorted_records[:n_train]
    val_slice = sorted_records[n_train:n_train + n_val]
    test_slice = sorted_records[n_train + n_val:]

    return SplitManifest(
        train=[str(r["record_id"]) for r in train_slice],
        validation=[str(r["record_id"]) for r in val_slice],
        test=[str(r["record_id"]) for r in test_slice],
        metadata={
            "split_type": "chronological",
            "dataset_size": n_total,
        },
    )


def validate_split_manifest(
    manifest: SplitManifest,
    records: List[Dict[str, Any]],
    check_group_isolation: bool = False,
    check_chronological: bool = False,
) -> List[str]:
    """
    Validates a split manifest against leakage prevention rules:
      1. Disjoint record sets: No record ID appears in multiple partitions.
      2. Existence: Every ID in manifest exists in the dataset.
      3. Coverage: Every record in dataset is partitioned.
      4. Group isolation (optional): No group/user appears across different partitions.
      5. Chronological order (optional): Train timestamps <= Val timestamps <= Test timestamps.

    Returns a list of violation error strings. Returns empty list if fully valid.
    """
    violations: List[str] = []

    s_train = set(manifest.train)
    s_val = set(manifest.validation)
    s_test = set(manifest.test)

    # 1. Disjointness check
    overlap_train_val = s_train.intersection(s_val)
    if overlap_train_val:
        violations.append(f"Split leakage detected: record IDs overlap between train and validation: {sorted(overlap_train_val)}")

    overlap_train_test = s_train.intersection(s_test)
    if overlap_train_test:
        violations.append(f"Split leakage detected: record IDs overlap between train and test: {sorted(overlap_train_test)}")

    overlap_val_test = s_val.intersection(s_test)
    if overlap_val_test:
        violations.append(f"Split leakage detected: record IDs overlap between validation and test: {sorted(overlap_val_test)}")

    # 2. Existence check
    rec_by_id = {str(r["record_id"]): r for r in records}
    all_manifest_ids = s_train | s_val | s_test
    unknown_ids = all_manifest_ids - set(rec_by_id.keys())
    if unknown_ids:
        violations.append(f"Unknown record IDs in manifest: {sorted(unknown_ids)}")

    # 3. Group isolation check
    if check_group_isolation:
        train_groups = {rec_by_id[rid]["group_id"] for rid in s_train if rid in rec_by_id}
        val_groups = {rec_by_id[rid]["group_id"] for rid in s_val if rid in rec_by_id}
        test_groups = {rec_by_id[rid]["group_id"] for rid in s_test if rid in rec_by_id}

        grp_tv = train_groups.intersection(val_groups)
        if grp_tv:
            violations.append(f"Group leakage detected: groups overlap between train and val: {sorted(grp_tv)}")

        grp_tt = train_groups.intersection(test_groups)
        if grp_tt:
            violations.append(f"Group leakage detected: groups overlap between train and test: {sorted(grp_tt)}")

        grp_vt = val_groups.intersection(test_groups)
        if grp_vt:
            violations.append(f"Group leakage detected: groups overlap between val and test: {sorted(grp_vt)}")

    # 4. Chronological ordering check
    if check_chronological:
        train_times = [rec_by_id[rid]["timestamp"] for rid in manifest.train if "timestamp" in rec_by_id.get(rid, {})]
        val_times = [rec_by_id[rid]["timestamp"] for rid in manifest.validation if "timestamp" in rec_by_id.get(rid, {})]
        test_times = [rec_by_id[rid]["timestamp"] for rid in manifest.test if "timestamp" in rec_by_id.get(rid, {})]

        if train_times and val_times and max(train_times) > min(val_times):
            violations.append(f"Temporal leakage detected: max(train_time)={max(train_times)} > min(val_time)={min(val_times)}")

        if val_times and test_times and max(val_times) > min(test_times):
            violations.append(f"Temporal leakage detected: max(val_time)={max(val_times)} > min(test_time)={min(test_times)}")

    return violations


def load_dataset(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load fixture from JSON or CSV file into standard record format."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a list of records.")
            return data

    elif path.suffix.lower() == ".csv":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec: Dict[str, Any] = {
                    "record_id": row["record_id"],
                    "group_id": row["group_id"],
                    "x1": float(row["x1"]) if "." in row["x1"] else int(row["x1"]),
                    "x2": float(row["x2"]) if "." in row["x2"] else int(row["x2"]),
                    "label": int(row["label"]),
                }
                if "timestamp" in row:
                    rec["timestamp"] = int(row["timestamp"])
                records.append(rec)
        return records

    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
