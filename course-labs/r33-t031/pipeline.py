"""
Course Lab R33-T031: Image Data Contract Validator & tf.data Preprocessing Pipeline.

Student: Snehakumari Vinod Singh
Task: R33-T031 (Task 1 of 10)
Target Item: 13_loading_and_preprocessing_data.ipynb (The tf.data API)

Disclaimer:
The synthetic fixture (six 4x4 grayscale images) is strictly intended for CPU smoke testing
and pipeline schema verification. It is NOT an observation from any real-world dataset and
must not be used as evidence for real-world or clinical performance.
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
    Validates image dataset records against the R33-T031 data contract:
      - record_id: Unique identifier (non-empty string or integer).
      - group_id: Group identifier (non-empty string or integer).
      - pixels: 4x4 matrix (or 16-element numeric sequence) with values in range [0, 255].
      - label: Binary class identifier (0 or 1).

    Ensures full transparency by collecting specific failure reasons rather than
    silently dropping invalid records.
    """

    def __init__(
        self,
        expected_height: int = 4,
        expected_width: int = 4,
        min_pixel_value: float = 0.0,
        max_pixel_value: float = 255.0,
        allowed_labels: Optional[Set[int]] = None,
    ) -> None:
        self.expected_height = expected_height
        self.expected_width = expected_width
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.allowed_labels = allowed_labels if allowed_labels is not None else {0, 1}

    def validate_record_id(
        self,
        record_id: Any,
        seen_ids: Optional[Set[Any]] = None,
    ) -> List[ValidationError]:
        """Validate record_id presence, type, non-emptiness, and uniqueness."""
        errors: List[ValidationError] = []

        if record_id is None:
            errors.append(ValidationError(
                record_id=None,
                field="record_id",
                reason="Missing record_id: field is None or undefined.",
            ))
            return errors

        if isinstance(record_id, str):
            if not record_id.strip():
                errors.append(ValidationError(
                    record_id=record_id,
                    field="record_id",
                    reason="Empty record_id: string must not be empty or whitespace only.",
                ))
        elif isinstance(record_id, int):
            pass
        else:
            errors.append(ValidationError(
                record_id=record_id,
                field="record_id",
                reason=f"Invalid record_id type: expected str or int, got {type(record_id).__name__}.",
            ))

        if seen_ids is not None:
            if record_id in seen_ids:
                errors.append(ValidationError(
                    record_id=record_id,
                    field="record_id",
                    reason=f"Duplicate record_id: '{record_id}' was already encountered in the dataset.",
                ))
            else:
                seen_ids.add(record_id)

        return errors

    def validate_group_id(self, record_id: Any, group_id: Any) -> List[ValidationError]:
        """Validate group_id presence and non-emptiness."""
        errors: List[ValidationError] = []

        if group_id is None:
            errors.append(ValidationError(
                record_id=record_id,
                field="group_id",
                reason="Missing group_id: field is None or undefined.",
            ))
            return errors

        if isinstance(group_id, str):
            if not group_id.strip():
                errors.append(ValidationError(
                    record_id=record_id,
                    field="group_id",
                    reason="Empty group_id: string must not be empty or whitespace only.",
                ))
        elif isinstance(group_id, int):
            pass
        else:
            errors.append(ValidationError(
                record_id=record_id,
                field="group_id",
                reason=f"Invalid group_id type: expected str or int, got {type(group_id).__name__}.",
            ))

        return errors

    def validate_label(self, record_id: Any, label: Any) -> List[ValidationError]:
        """Validate label presence, integer representation, and domain constraints."""
        errors: List[ValidationError] = []

        if label is None:
            errors.append(ValidationError(
                record_id=record_id,
                field="label",
                reason="Missing label: field is None or undefined.",
            ))
            return errors

        if isinstance(label, bool):
            errors.append(ValidationError(
                record_id=record_id,
                field="label",
                reason=f"Malformed label: boolean value {label} is not permitted as integer class label.",
            ))
            return errors

        if not isinstance(label, int):
            try:
                converted = float(label)
                if not converted.is_integer():
                    errors.append(ValidationError(
                        record_id=record_id,
                        field="label",
                        reason=f"Malformed label: non-integer float value '{label}'.",
                    ))
                    return errors
                label = int(converted)
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    record_id=record_id,
                    field="label",
                    reason=f"Malformed label: unable to parse '{label}' as integer class label.",
                ))
                return errors

        if label not in self.allowed_labels:
            errors.append(ValidationError(
                record_id=record_id,
                field="label",
                reason=f"Label out of range: value {label} is not in allowed classes {sorted(self.allowed_labels)}.",
            ))

        return errors

    def validate_pixels(self, record_id: Any, pixels: Any) -> List[ValidationError]:
        """Validate 4x4 image tensor / pixels shape, numeric data types, and value range [0, 255]."""
        errors: List[ValidationError] = []

        if pixels is None:
            errors.append(ValidationError(
                record_id=record_id,
                field="pixels",
                reason="Missing pixels: field is None or undefined.",
            ))
            return errors

        if not isinstance(pixels, (list, tuple)):
            errors.append(ValidationError(
                record_id=record_id,
                field="pixels",
                reason=f"Malformed tensor: expected list or sequence of pixels, got {type(pixels).__name__}.",
            ))
            return errors

        # Case A: 2D matrix (4 rows)
        if len(pixels) == self.expected_height and all(isinstance(row, (list, tuple)) for row in pixels):
            for row_idx, row in enumerate(pixels):
                if len(row) != self.expected_width:
                    errors.append(ValidationError(
                        record_id=record_id,
                        field="pixels",
                        reason=f"Malformed tensor shape: row {row_idx} has length {len(row)}, expected {self.expected_width}.",
                    ))
                    return errors
                for col_idx, val in enumerate(row):
                    val_err = self._check_pixel_value(record_id, val, f"pixels[{row_idx}][{col_idx}]")
                    if val_err:
                        errors.append(val_err)
            return errors

        # Case B: Flat 1D sequence of length (height * width)
        flat_expected = self.expected_height * self.expected_width
        if len(pixels) == flat_expected and not any(isinstance(x, (list, tuple)) for x in pixels):
            for idx, val in enumerate(pixels):
                val_err = self._check_pixel_value(record_id, val, f"pixels[{idx}]")
                if val_err:
                    errors.append(val_err)
            return errors

        # Invalid shape
        errors.append(ValidationError(
            record_id=record_id,
            field="pixels",
            reason=f"Malformed tensor dimensions: expected {self.expected_height}x{self.expected_width} 2D matrix or {flat_expected}-element flat array, got sequence of length {len(pixels)}.",
        ))
        return errors

    def _check_pixel_value(self, record_id: Any, val: Any, loc: str) -> Optional[ValidationError]:
        """Validate an individual pixel numeric value."""
        if val is None or isinstance(val, bool):
            return ValidationError(
                record_id=record_id,
                field="pixels",
                reason=f"Malformed numeric at {loc}: expected numeric value, got {val!r}.",
            )

        if not isinstance(val, (int, float)):
            return ValidationError(
                record_id=record_id,
                field="pixels",
                reason=f"Malformed numeric at {loc}: non-numeric value '{val}' ({type(val).__name__}).",
            )

        if math.isnan(val) or math.isinf(val):
            return ValidationError(
                record_id=record_id,
                field="pixels",
                reason=f"Malformed numeric at {loc}: NaN or infinite value.",
            )

        if val < self.min_pixel_value or val > self.max_pixel_value:
            return ValidationError(
                record_id=record_id,
                field="pixels",
                reason=f"Pixel value out of range at {loc}: {val} not in [{self.min_pixel_value}, {self.max_pixel_value}].",
            )

        return None

    def validate_record(
        self,
        record: Any,
        seen_ids: Optional[Set[Any]] = None,
    ) -> List[ValidationError]:
        """Validate a single record dictionary against all contract rules."""
        if not isinstance(record, dict):
            return [ValidationError(
                record_id=None,
                field="record",
                reason=f"Malformed record: expected dict mapping, got {type(record).__name__}.",
            )]

        record_id = record.get("record_id")
        errors: List[ValidationError] = []

        errors.extend(self.validate_record_id(record_id, seen_ids=seen_ids))
        errors.extend(self.validate_group_id(record_id, record.get("group_id")))
        errors.extend(self.validate_pixels(record_id, record.get("pixels")))
        errors.extend(self.validate_label(record_id, record.get("label")))

        return errors

    def validate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        """
        Validate an entire list of records.

        Returns:
            Tuple of (valid_records, all_validation_errors).
            Guarantees no record is silently dropped; every dropped record is tracked
            with detailed errors in all_validation_errors.
        """
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


def load_dataset(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load dataset from JSON or CSV file into standard record format."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a top-level list of record objects.")
            return data

    elif path.suffix.lower() == ".csv":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pixels_raw = row["pixels"]
                pixels = json.loads(pixels_raw)
                records.append({
                    "record_id": row["record_id"],
                    "group_id": row["group_id"],
                    "pixels": pixels,
                    "label": int(row["label"]),
                })
        return records

    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Use .json or .csv.")


def fixed_stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic stratified train/test split preserving label distribution.
    For small smoke fixtures, achieves exact class balance without shuffling jitter.
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
