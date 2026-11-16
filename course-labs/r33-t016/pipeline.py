import math
import numpy as np


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    record_ids = set()
    group_ids = set()

    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each record must be a dictionary")

        record_id = row.get("record_id")
        group_id = row.get("group_id")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")

        if record_id in record_ids:
            raise ValueError("duplicate record_id")
        if group_id in group_ids:
            raise ValueError("duplicate group_id")

        record_ids.add(record_id)
        group_ids.add(group_id)

        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("label must be integer 0 or 1")

        pixels = row.get("pixels")
        if not isinstance(pixels, list) or len(pixels) != 4:
            raise ValueError("image must have exactly four rows")

        for image_row in pixels:
            if not isinstance(image_row, list) or len(image_row) != 4:
                raise ValueError("each image row must have four pixels")

            for pixel in image_row:
                if isinstance(pixel, bool) or not isinstance(
                    pixel, (int, float)
                ):
                    raise ValueError("pixels must be numeric")
                if not math.isfinite(pixel) or not 0 <= pixel <= 255:
                    raise ValueError("pixels must be finite and in 0..255")

    return True


def validate_splits(records, splits):
    if not isinstance(splits, dict) or set(splits) != {
        "train", "validation", "test"
    }:
        raise ValueError("splits must contain train, validation, and test")

    known_ids = {row["record_id"] for row in records}
    seen_ids = set()
    group_by_id = {row["record_id"]: row["group_id"] for row in records}

    for split_name in ("train", "validation", "test"):
        ids = splits[split_name]

        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split_name} split must be non-empty")

        for record_id in ids:
            if record_id not in known_ids:
                raise ValueError("split contains an unknown record ID")
            if record_id in seen_ids:
                raise ValueError("split overlap detected")
            seen_ids.add(record_id)

    if seen_ids != known_ids:
        raise ValueError("every record must appear in exactly one split")

    split_groups = {}
    for split_name, ids in splits.items():
        split_groups[split_name] = {
            group_by_id[record_id] for record_id in ids
        }

    if split_groups["train"] & split_groups["validation"]:
        raise ValueError("training and validation groups overlap")
    if split_groups["train"] & split_groups["test"]:
        raise ValueError("training and test groups overlap")
    if split_groups["validation"] & split_groups["test"]:
        raise ValueError("validation and test groups overlap")

    return True


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def run_pipeline(fixture):
    records = fixture["records"]
    splits = fixture["splits"]
    seed = fixture["seed"]
    max_samples = fixture["max_samples"]
    max_updates = fixture["max_updates"]

    validate_records(records)
    validate_splits(records, splits)

    if len(records) > 32 or len(records) > max_samples:
        raise ValueError("sample limit exceeded")
    if type(max_updates) is not int or not 1 <= max_updates <= 20:
        raise ValueError("updates must be between 1 and 20")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    by_id = {row["record_id"]: row for row in records}

    images = np.asarray(
        [row["pixels"] for row in records], dtype=np.float32
    ) / np.float32(255.0)

    x = images.reshape(len(records), 16)
    y = np.asarray([row["label"] for row in records], dtype=np.int64)

    train_indices = [
        next(i for i, row in enumerate(records) if row["record_id"] == rid)
        for rid in splits["train"]
    ]

    x_train = x[train_indices]
    y_train = y[train_indices]

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(16, 2)).astype(np.float32)
    bias = np.zeros((1, 2), dtype=np.float32)

    learning_rate = np.float32(0.5)
    final_loss = float("nan")

    for _ in range(max_updates):
        probabilities = softmax(x_train @ weights + bias)
        targets = np.eye(2, dtype=np.float32)[y_train]

        final_loss = float(
            -np.mean(
                np.sum(
                    targets * np.log(np.clip(probabilities, 1e-7, 1.0)),
                    axis=1
                )
            )
        )

        gradient = (probabilities - targets) / len(y_train)
        weights -= learning_rate * (x_train.T @ gradient)
        bias -= learning_rate * np.sum(gradient, axis=0, keepdims=True)

    all_probabilities = softmax(x @ weights + bias)
    predictions = np.argmax(all_probabilities, axis=1)

    if not np.isfinite(final_loss):
        raise ValueError("training loss is not finite")
    if not np.all(np.isfinite(all_probabilities)):
        raise ValueError("predictions contain non-finite values")

    return {
        "model": "small NumPy softmax CPU smoke classifier",
        "model_type": "synthetic smoke model, not a production model",
        "seed": seed,
        "sample_count": len(records),
        "training_sample_count": len(train_indices),
        "step_count": max_updates,
        "loss": final_loss,
        "loss_is_finite": bool(np.isfinite(final_loss)),
        "prediction_shape": list(all_probabilities.shape),
        "predictions": predictions.tolist(),
        "probabilities": all_probabilities.tolist(),
        "status": "synthetic pipeline verification only"
    }
