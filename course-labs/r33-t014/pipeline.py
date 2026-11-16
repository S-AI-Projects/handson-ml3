"""Small, standard-library-only evaluation metric checks."""

from math import isfinite


def validate_records(records):
    errors = []
    seen_ids = set()
    seen_groups = set()

    if not isinstance(records, list):
        return [{"field": "records", "reason": "must be a list"}]

    for index, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append({
                "field": "record",
                "reason": f"row {index} must be an object"
            })
            continue

        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append({
                "field": "record_id",
                "reason": "must be a non-empty string"
            })
        elif record_id in seen_ids:
            errors.append({
                "field": "record_id",
                "reason": "duplicate record ID"
            })
        else:
            seen_ids.add(record_id)

        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            errors.append({
                "field": "group_id",
                "reason": "must be a non-empty string"
            })
        elif group_id in seen_groups:
            errors.append({
                "field": "group_id",
                "reason": "group IDs must be unique in this fixture"
            })
        else:
            seen_groups.add(group_id)

        for field in ("x1", "x2"):
            value = row.get(field)
            if type(value) not in (int, float) or not isfinite(value):
                errors.append({
                    "field": field,
                    "reason": "must be a finite number"
                })

        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            errors.append({
                "field": "label",
                "reason": "must be integer 0 or 1"
            })

    return errors


def validate_splits(records, splits):
    errors = []
    if not isinstance(splits, dict):
        return [{"field": "splits", "reason": "must be an object"}]

    required = ("train", "validation", "test")
    if any(name not in splits for name in required):
        return [{"field": "splits", "reason": "train, validation and test are required"}]

    known_ids = {
        row["record_id"] for row in records
        if isinstance(row, dict) and isinstance(row.get("record_id"), str)
    }

    seen = set()
    for name in required:
        ids = splits[name]
        if not isinstance(ids, list):
            errors.append({"field": name, "reason": "must be a list"})
            continue

        for record_id in ids:
            if record_id not in known_ids:
                errors.append({
                    "field": name,
                    "reason": f"unknown record ID: {record_id}"
                })
            if record_id in seen:
                errors.append({
                    "field": name,
                    "reason": f"record appears in more than one split: {record_id}"
                })
            seen.add(record_id)

    if seen != known_ids:
        errors.append({
            "field": "splits",
            "reason": "every record must appear in exactly one split"
        })

    group_by_id = {
        row["record_id"]: row.get("group_id")
        for row in records
        if isinstance(row, dict) and isinstance(row.get("record_id"), str)
    }

    split_groups = {}
    for name in required:
        split_groups[name] = {
            group_by_id[record_id]
            for record_id in splits[name]
            if record_id in group_by_id
        }

    for i, first in enumerate(required):
        for second in required[i + 1:]:
            if split_groups[first] & split_groups[second]:
                errors.append({
                    "field": "group_id",
                    "reason": f"group leakage between {first} and {second}"
                })

    return errors


def classification_metrics(truth, prediction):
    """Binary positive-class metrics; zero denominator returns 0.0."""
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have equal non-zero lengths")

    if any(type(value) is not int or value not in (0, 1)
           for value in truth + prediction):
        raise ValueError("classification values must be integer 0 or 1")

    true_positive = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip(truth, prediction)
    )
    false_positive = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip(truth, prediction)
    )
    false_negative = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip(truth, prediction)
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "averaging": "binary positive class (label 1)",
        "zero_denominator": 0.0
    }


def mean_absolute_error(truth, prediction):
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have equal non-zero lengths")

    if any(type(value) not in (int, float) or not isfinite(value)
           for value in truth + prediction):
        raise ValueError("regression values must be finite numbers")

    return sum(
        abs(actual - predicted)
        for actual, predicted in zip(truth, prediction)
    ) / len(truth)


def run_evaluation(data):
    record_errors = validate_records(data["records"])
    split_errors = validate_splits(data["records"], data["splits"])

    if record_errors or split_errors:
        raise ValueError({
            "record_errors": record_errors,
            "split_errors": split_errors
        })

    classification = classification_metrics(
        data["classification_example"]["truth"],
        data["classification_example"]["prediction"]
    )

    regression = mean_absolute_error(
        data["regression_example"]["truth"],
        data["regression_example"]["prediction"]
    )

    return {
        "classification": classification,
        "regression_mae": regression,
        "fixture_record_count": len(data["records"]),
        "fixture_status": "synthetic pipeline verification only"
    }
