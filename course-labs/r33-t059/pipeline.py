"""Image-interface pipeline over compressed TFRecords with a robustness check.

Documented result (the data contract the test protects):
  * per-record prediction keyed by record_id, from a fixed rule:
    predict 1 if any pixel > 0 else 0
  * overall accuracy against the stored labels
  * per-class record counts

Robustness property tested: record ORDER changes are irrelevant when records
carry stable IDs - after aligning by record_id the result is identical.
Time order is not applicable to this image fixture and is never randomised.

Unsupported inputs (documented behaviour):
  * missing field            -> rejected (SchemaError)
  * shape mismatch           -> rejected (SchemaError)
  * unseen category/label    -> mapped to FALLBACK_LABEL (-1), reported
This is a narrow property check, not a claim of broad adversarial robustness.
"""
import random

from fixture import FALLBACK_LABEL, H, KNOWN_LABELS, SCHEMA, W
from tfrecord_io import read_tfrecord_gz, write_tfrecord_gz


class SchemaError(ValueError):
    pass


def to_example(rec):
    return {
        "record_id": rec["record_id"],
        "group_id": rec["group_id"],
        "pixels": list(rec["pixels"]),
        "label": int(rec["label"]),
    }


def validate(raw):
    """Return a clean record or raise SchemaError / map unseen label."""
    missing = [f for f in SCHEMA if f not in raw]
    if missing:
        raise SchemaError("missing field(s): %s" % ", ".join(missing))
    pixels = raw["pixels"]
    if len(pixels) != H * W:
        raise SchemaError(
            "shape mismatch: expected %d pixels (%dx%d), got %d"
            % (H * W, H, W, len(pixels))
        )
    if any(not 0 <= p <= 255 for p in pixels):
        raise SchemaError("pixel value outside 0..255")
    label_list = raw["label"]
    label = label_list[0] if isinstance(label_list, list) else label_list
    fallback = label not in KNOWN_LABELS
    return {
        "record_id": raw["record_id"].decode() if isinstance(raw["record_id"], bytes) else raw["record_id"],
        "group_id": raw["group_id"].decode() if isinstance(raw["group_id"], bytes) else raw["group_id"],
        "pixels": list(pixels),
        "label": FALLBACK_LABEL if fallback else label,
        "label_fallback": fallback,
    }


def predict(pixels):
    return 1 if any(p > 0 for p in pixels) else 0


def summarise(raw_records):
    """Validate every record and compute the documented, order-free result."""
    records = [validate(r) for r in raw_records]
    ids = [r["record_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SchemaError("duplicate record_id")
    by_id = {r["record_id"]: r for r in records}
    preds = {i: predict(by_id[i]["pixels"]) for i in sorted(by_id)}
    scored = [r for r in records if not r["label_fallback"]]
    correct = sum(preds[r["record_id"]] == r["label"] for r in scored)
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {
        "predictions": preds,
        "accuracy": correct / len(scored) if scored else None,
        "class_counts": dict(sorted(counts.items())),
        "fallback_ids": sorted(r["record_id"] for r in records if r["label_fallback"]),
    }


def roundtrip(path, records):
    write_tfrecord_gz(path, [to_example(r) for r in records])
    return read_tfrecord_gz(path)


def shuffled(records, seed=0):
    out = list(records)
    random.Random(seed).shuffle(out)
    return out
