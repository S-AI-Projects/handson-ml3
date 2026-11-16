"""Minimal majority-class image classification baseline."""


def validate_records(records):
    errors = []
    seen_ids = set()

    if not isinstance(records, list):
        return [{
            "record_id": "<dataset>",
            "field": "records",
            "reason": "must be a list"
        }]

    for index, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append({
                "record_id": f"<row-{index}>",
                "field": "record",
                "reason": "must be an object"
            })
            continue

        record_id = row.get("record_id")

        if not isinstance(record_id, str) or not record_id.strip():
            errors.append({
                "record_id": record_id if record_id is not None else "<missing>",
                "field": "record_id",
                "reason": "must be a non-empty string"
            })
        elif record_id in seen_ids:
            errors.append({
                "record_id": record_id,
                "field": "record_id",
                "reason": "duplicate record ID"
            })
        else:
            seen_ids.add(record_id)

        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            errors.append({
                "record_id": record_id,
                "field": "group_id",
                "reason": "must be a non-empty string"
            })

        pixels = row.get("pixels")
        valid_pixels = (
            isinstance(pixels, list)
            and len(pixels) == 4
            and all(isinstance(line, list) and len(line) == 4
                    for line in pixels)
        )

        if not valid_pixels:
            errors.append({
                "record_id": record_id,
                "field": "pixels",
                "reason": "must be a 4x4 matrix"
            })
        else:
            for line in pixels:
                for value in line:
                    if type(value) is not int or value < 0 or value > 255:
                        errors.append({
                            "record_id": record_id,
                            "field": "pixels",
                            "reason": "pixel values must be integers from 0 to 255"
                        })
                        break
                else:
                    continue
                break

        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            errors.append({
                "record_id": record_id,
                "field": "label",
                "reason": "must be integer 0 or 1"
            })

    return errors





def validate_splits(records, splits):
    errors = []
    by_id = {row["record_id"]: row for row in records
             if isinstance(row, dict) and isinstance(row.get("record_id"), str)}

    required = ("train", "validation", "test")

    for name in required:
        if name not in splits or not isinstance(splits[name], list):
            errors.append({
                "record_id": "<splits>",
                "field": name,
                "reason": "split must be a list"
            })

    if errors:
        return errors

    seen_records = set()
    seen_groups = {}

    for name in required:
        for record_id in splits[name]:
            if record_id not in by_id:
                errors.append({
                    "record_id": record_id,
                    "field": name,
                    "reason": "unknown record ID"
                })
                continue

            if record_id in seen_records:
                errors.append({
                    "record_id": record_id,
                    "field": name,
                    "reason": "record appears in more than one split"
                })
            seen_records.add(record_id)

            group_id = by_id[record_id].get("group_id")
            if group_id in seen_groups and seen_groups[group_id] != name:
                errors.append({
                    "record_id": record_id,
                    "field": "group_id",
                    "reason": "group appears in more than one split"
                })
            else:
                seen_groups[group_id] = name

    if seen_records != set(by_id):
        errors.append({
            "record_id": "<splits>",
            "field": "splits",
            "reason": "every record must appear in exactly one split"
        })

    return errors


def fit_majority_baseline(records, train_ids):
    by_id = {row["record_id"]: row for row in records}
    if not train_ids:
        raise ValueError("training split must not be empty")

    counts = {}
    for record_id in train_ids:
        if record_id not in by_id:
            raise ValueError(f"unknown training record: {record_id}")
        label = by_id[record_id]["label"]
        counts[label] = counts.get(label, 0) + 1

    # Deterministic tie-break: choose the smaller label.
    return min(counts, key=lambda label: (-counts[label], label))


def calculate_accuracy(records, predictions, test_ids):
    by_id = {row["record_id"]: row for row in records}
    if not test_ids:
        raise ValueError("test split must not be empty")

    correct = sum(
        predictions[record_id] == by_id[record_id]["label"]
        for record_id in test_ids
    )
    return correct / len(test_ids)


def run_experiment(data):
    records = data["records"]
    splits = data["splits"]

    record_errors = validate_records(records)
    split_errors = validate_splits(records, splits)
    errors = record_errors + split_errors

    if errors:
        raise ValueError(str(errors))

    majority_label = fit_majority_baseline(records, splits["train"])

    predictions = {
        record_id: majority_label
        for record_id in splits["test"]
    }

    test_accuracy = calculate_accuracy(
        records, predictions, splits["test"]
    )

    return {
        "baseline": "majority_class",
        "training_majority_label": majority_label,
        "test_predictions": predictions,
        "test_ids": splits["test"],
        "test_accuracy": test_accuracy
    }
