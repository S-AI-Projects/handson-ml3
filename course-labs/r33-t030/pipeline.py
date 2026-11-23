"""Schema validator for classification input fields for R33-T030.

Validates: record_id, group_id, x1, x2, label.
Reports record ID, field, and specific failure reason without silently dropping records.
"""
from typing import List, Dict, Any

REQUIRED_FIELDS = ["record_id", "group_id", "x1", "x2", "label"]

class ClassificationSchemaValidator:
    """Validates records against the tabular classification schema contract."""
    def __init__(self) -> None:
        self.seen_ids = set()

    def validate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validates all records without silently dropping any."""
        failures = []
        validated_records = []
        seen_ids = set()

        for idx, r in enumerate(records):
            rec_id = r.get("record_id")
            has_error = False

            # Check for missing record_id
            if rec_id is None or str(rec_id).strip() == "":
                failures.append({
                    "record_id": f"<index_{idx}>",
                    "field": "record_id",
                    "reason": "Missing or empty record_id"
                })
                has_error = True
            elif rec_id in seen_ids:
                failures.append({
                    "record_id": rec_id,
                    "field": "record_id",
                    "reason": f"Duplicate record_id '{rec_id}' encountered"
                })
                has_error = True
            else:
                seen_ids.add(rec_id)

            # Check other required fields
            for field in ["group_id", "x1", "x2", "label"]:
                if field not in r:
                    failures.append({
                        "record_id": rec_id or f"<index_{idx}>",
                        "field": field,
                        "reason": f"Missing required field '{field}'"
                    })
                    has_error = True

            # Validate field types and constraints if present
            if "x1" in r and not isinstance(r["x1"], (int, float)):
                failures.append({
                    "record_id": rec_id or f"<index_{idx}>",
                    "field": "x1",
                    "reason": f"Malformed numeric field 'x1': expected float/int, got {type(r['x1']).__name__}"
                })
                has_error = True

            if "x2" in r and not isinstance(r["x2"], (int, float)):
                failures.append({
                    "record_id": rec_id or f"<index_{idx}>",
                    "field": "x2",
                    "reason": f"Malformed numeric field 'x2': expected float/int, got {type(r['x2']).__name__}"
                })
                has_error = True

            if "group_id" in r and not isinstance(r["group_id"], str):
                failures.append({
                    "record_id": rec_id or f"<index_{idx}>",
                    "field": "group_id",
                    "reason": f"Malformed text field 'group_id': expected str, got {type(r['group_id']).__name__}"
                })
                has_error = True

            if "label" in r and r["label"] not in (0, 1):
                failures.append({
                    "record_id": rec_id or f"<index_{idx}>",
                    "field": "label",
                    "reason": f"Malformed binary label: expected 0 or 1, got {r['label']}"
                })
                has_error = True

            # Always preserve every record in the processed output
            validated_records.append({
                "record": r,
                "is_valid": not has_error
            })

        return {
            "total_processed": len(records),
            "is_valid": len(failures) == 0,
            "failure_count": len(failures),
            "failures": failures,
            "processed_records": validated_records
        }
