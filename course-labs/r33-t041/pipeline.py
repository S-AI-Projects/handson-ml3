"""
Course Lab R33-T041: Time Series Data Contract Validator
Enforces input schema (record_id, timestamp, value) for Basic RNN time series observations.
"""
import re
import math
from typing import Dict, Any, List, Tuple, Optional, Set

REQUIRED_FIELDS = ("record_id", "timestamp", "value")
ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def validate_record(record: Any, seen_ids: Optional[Set[str]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validates an individual time series record without silently dropping records.
    Returns (True, None) if valid, or (False, error_dict) if invalid.
    """
    if not isinstance(record, dict):
        return False, {
            "record_id": None,
            "field": "record",
            "reason": f"Record must be a dictionary, got {type(record).__name__}"
        }

    rec_id = record.get("record_id")
    if rec_id is None or (isinstance(rec_id, str) and not rec_id.strip()):
        return False, {
            "record_id": rec_id,
            "field": "record_id",
            "reason": "Missing or empty record_id"
        }

    str_rec_id = str(rec_id)
    if seen_ids is not None:
        if str_rec_id in seen_ids:
            return False, {
                "record_id": rec_id,
                "field": "record_id",
                "reason": f"Duplicate record_id: {rec_id}"
            }
        seen_ids.add(str_rec_id)

    timestamp = record.get("timestamp")
    if timestamp is None:
        return False, {
            "record_id": rec_id,
            "field": "timestamp",
            "reason": "Missing required field: timestamp"
        }
    if not isinstance(timestamp, str) or not ISO_DATE_REGEX.match(timestamp):
        return False, {
            "record_id": rec_id,
            "field": "timestamp",
            "reason": f"Malformed timestamp '{timestamp}'; expected YYYY-MM-DD string"
        }

    if "value" not in record or record.get("value") is None:
        return False, {
            "record_id": rec_id,
            "field": "value",
            "reason": "Missing required field: value"
        }

    val = record["value"]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False, {
            "record_id": rec_id,
            "field": "value",
            "reason": f"Malformed numeric value: expected scalar int/float, got {type(val).__name__}"
        }
    if math.isnan(val) or math.isinf(val):
        return False, {
            "record_id": rec_id,
            "field": "value",
            "reason": f"Malformed numeric value: {val} (NaN or Inf not permitted)"
        }

    return True, None

def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates a list of time series records.
    Reports all passing and failing records without silently discarding any record.
    """
    seen_ids: Set[str] = set()
    valid_records = []
    failures = []

    for idx, rec in enumerate(records):
        passed, err = validate_record(rec, seen_ids=seen_ids)
        if passed:
            valid_records.append(rec)
        else:
            if err:
                err["record_index"] = idx
            failures.append(err)

    return {
        "valid_records": valid_records,
        "failures": failures,
        "summary": {
            "total": len(records),
            "passed": len(valid_records),
            "failed": len(failures)
        }
    }

def partition_time_series(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Partitions 10 daily observations: first 6 train, next 2 validation, final 2 test.
    Preserves strict chronological order without shuffling.
    """
    if len(records) != 10:
        raise ValueError(f"Expected exactly 10 records for standard fixture, got {len(records)}")
    return {
        "train": records[:6],
        "validation": records[6:8],
        "test": records[8:10]
    }
