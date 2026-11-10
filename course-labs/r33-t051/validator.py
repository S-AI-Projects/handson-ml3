"""Schema validator for text input fields: record_id, text, label.

Never silently drops a record: every problem is reported as a Failure
with the record ID, the field, and a specific reason.
"""
from dataclasses import dataclass

FIELDS = ("record_id", "text", "label")


@dataclass(frozen=True)
class Failure:
    record_id: object
    field: str
    reason: str


def validate_records(records):
    failures = []
    seen_ids = set()

    for index, rec in enumerate(records):
        placeholder = f"<index {index}>"

        if not isinstance(rec, dict):
            failures.append(Failure(placeholder, "record", "record is not a mapping"))
            continue

        # record_id
        rid = rec.get("record_id")
        if rid is None or (isinstance(rid, str) and not rid.strip()):
            failures.append(Failure(placeholder, "record_id", "missing record_id"))
            shown = placeholder
        elif not isinstance(rid, str):
            failures.append(Failure(placeholder, "record_id", "record_id must be a string"))
            shown = placeholder
        else:
            shown = rid
            if rid in seen_ids:
                failures.append(Failure(rid, "record_id", "duplicate record_id"))
            seen_ids.add(rid)

        # text
        text = rec.get("text")
        if text is None:
            failures.append(Failure(shown, "text", "missing text"))
        elif not isinstance(text, str):
            failures.append(Failure(shown, "text", f"text must be str, got {type(text).__name__} (malformed text/tensor data)"))
        elif not text.strip():
            failures.append(Failure(shown, "text", "text is empty"))

        # label
        label = rec.get("label")
        if label is None:
            failures.append(Failure(shown, "label", "missing label"))
        elif isinstance(label, bool) or not isinstance(label, int):
            failures.append(Failure(shown, "label", f"label must be int 0/1, got {type(label).__name__} (malformed numeric data)"))
        elif label not in (0, 1):
            failures.append(Failure(shown, "label", f"label {label} out of range, expected 0 or 1"))

        # unexpected fields
        for key in rec:
            if key not in FIELDS:
                failures.append(Failure(shown, key, "unexpected field"))

    return failures
