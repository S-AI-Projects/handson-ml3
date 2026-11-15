"""Validation utilities for the R33-T011 regression data contract."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

REQUIRED_FIELDS = ("record_id", "x1", "x2", "target")
NUMERIC_FIELDS = ("x1", "x2", "target")


@dataclass(frozen=True)
class ValidationError:
    record_id: str
    field: str
    reason: str


def validate_records(records):
    """Validate records and return every detected contract violation."""
    errors = []
    seen_ids = set()

    if not isinstance(records, list):
        return [ValidationError(
            "<dataset>", "<records>", "Expected a list of records."
        )]

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(ValidationError(
                f"<row:{index}>", "<record>",
                "Expected a record mapping/dictionary."
            ))
            continue

        raw_id = record.get("record_id")
        record_id = (
            raw_id.strip()
            if isinstance(raw_id, str) and raw_id.strip()
            else "<missing>"
        )

        if "record_id" not in record:
            errors.append(ValidationError(
                record_id, "record_id", "Required field is missing."
            ))
        elif not isinstance(raw_id, str) or not raw_id.strip():
            errors.append(ValidationError(
                record_id, "record_id", "Expected a non-empty string."
            ))
        elif record_id in seen_ids:
            errors.append(ValidationError(
                record_id, "record_id", "Duplicate record ID."
            ))
        else:
            seen_ids.add(record_id)

        for field in NUMERIC_FIELDS:
            if field not in record:
                errors.append(ValidationError(
                    record_id, field, "Required field is missing."
                ))
                continue

            value = record[field]

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(ValidationError(
                    record_id, field,
                    "Expected a numeric scalar; text, tensors and nested values are not allowed."
                ))
            elif not math.isfinite(value):
                errors.append(ValidationError(
                    record_id, field, "Numeric value must be finite."
                ))

        for field in record:
            if field not in REQUIRED_FIELDS:
                errors.append(ValidationError(
                    record_id, field,
                    "Unexpected field; not part of the data contract."
                ))

    return errors


def format_errors(errors):
    """Convert validation errors into JSON-friendly dictionaries."""
    return [
        {
            "record_id": error.record_id,
            "field": error.field,
            "reason": error.reason
        }
        for error in errors
    ]


def validate_fixture_splits(fixture):
    """Check the required deterministic train/validation/test split."""
    if not isinstance(fixture, Mapping):
        return [ValidationError(
            "<fixture>", "<fixture>",
            "Expected a fixture mapping/dictionary."
        )]

    records = fixture.get("records")
    splits = fixture.get("splits")

    if not isinstance(records, list):
        return [ValidationError(
            "<fixture>", "records", "Expected records to be a list."
        )]

    if not isinstance(splits, Mapping):
        return [ValidationError(
            "<fixture>", "splits",
            "Expected splits to be a mapping/dictionary."
        )]

    errors = []
    expected = {
        "train": ["r1", "r2", "r3", "r4"],
        "validation": ["r5"],
        "test": ["r6"]
    }

    if dict(splits) != expected:
        errors.append(ValidationError(
            "<fixture>", "splits",
            "Split assignments do not match the required design."
        ))

    record_ids = set()
    for record in records:
        if isinstance(record, Mapping):
            rid = record.get("record_id")
            if isinstance(rid, str) and rid.strip():
                record_ids.add(rid.strip())

    flattened = []
    for split_name, ids in splits.items():
        if not isinstance(ids, list):
            errors.append(ValidationError(
                "<fixture>", str(split_name),
                "Each split must contain a list of record IDs."
            ))
            continue

        for rid in ids:
            if not isinstance(rid, str):
                errors.append(ValidationError(
                    "<fixture>", str(split_name),
                    "Split IDs must be strings."
                ))
            else:
                flattened.append(rid)

    if len(flattened) != len(set(flattened)):
        errors.append(ValidationError(
            "<fixture>", "splits",
            "A record ID appears in more than one split."
        ))

    if set(flattened) != record_ids:
        errors.append(ValidationError(
            "<fixture>", "splits",
            "Split IDs must cover every fixture record exactly once."
        ))

    return errors
