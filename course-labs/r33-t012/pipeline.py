import json
import random
from pathlib import Path


SPLITS = ("train", "validation", "test")


def load_fixture(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("Fixture must contain a records list.")

    return data["records"]


def validate_records(records):
    required = {"record_id", "group_id", "x1", "x2", "label"}
    seen_ids = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} must be an object.")

        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"Record at index {index} is missing fields: {sorted(missing)}"
            )

        for field in ("record_id", "group_id"):
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string.")

        if record["record_id"] in seen_ids:
            raise ValueError(f"Duplicate record_id: {record['record_id']}")
        seen_ids.add(record["record_id"])

        for field in ("x1", "x2"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{record['record_id']}: {field} must be numeric."
                )

        label = record["label"]
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(
                f"{record['record_id']}: label must be 0 or 1."
            )

    if not records:
        raise ValueError("Dataset must not be empty.")

    return True


def validate_manifest(records, manifest):
    validate_records(records)

    if set(manifest) != set(SPLITS):
        raise ValueError("Manifest must contain train, validation, and test.")

    record_ids = {record["record_id"] for record in records}
    group_by_id = {
        record["record_id"]: record["group_id"] for record in records
    }

    seen_ids = set()
    seen_groups = set()

    for split in SPLITS:
        ids = manifest[split]

        if not isinstance(ids, list):
            raise ValueError(f"{split} must contain a list of record IDs.")

        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate record IDs inside {split}.")

        for record_id in ids:
            if record_id not in record_ids:
                raise ValueError(f"Unknown record ID: {record_id}")

            if record_id in seen_ids:
                raise ValueError(
                    f"Record leakage: {record_id} appears in multiple splits."
                )

            group_id = group_by_id[record_id]
            if group_id in seen_groups:
                raise ValueError(
                    f"Group leakage: {group_id} appears in multiple splits."
                )

            seen_ids.add(record_id)
            seen_groups.add(group_id)

    if seen_ids != record_ids:
        missing = sorted(record_ids - seen_ids)
        raise ValueError(f"Records missing from manifest: {missing}")

    return True


def create_manifest(records, seed=42):
    validate_records(records)

    # Refuse repeated groups rather than risk placing related rows
    # in different splits.
    groups = [record["group_id"] for record in records]
    if len(groups) != len(set(groups)):
        raise ValueError(
            "This row-level stratified splitter requires one record per group. "
            "Use a group-aware split for repeated groups."
        )

    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record["record_id"])

    manifest = {split: [] for split in SPLITS}

    for label in sorted(by_label):
        ids = sorted(by_label[label])
        random.Random(seed + int(label)).shuffle(ids)

        train_count = int(len(ids) * 0.50)
        validation_count = int(len(ids) * 0.25)

        manifest["train"].extend(ids[:train_count])
        manifest["validation"].extend(
            ids[train_count:train_count + validation_count]
        )
        manifest["test"].extend(
            ids[train_count + validation_count:]
        )

    for split in SPLITS:
        manifest[split].sort()

    validate_manifest(records, manifest)
    return manifest


if __name__ == "__main__":
    fixture_path = Path(__file__).with_name("fixture.json")
    rows = load_fixture(fixture_path)
    result = create_manifest(rows)
    print(json.dumps(result, indent=2))
