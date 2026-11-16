"""Explicit, leakage-resistant preprocessing for synthetic grayscale images."""

import math

import numpy as np


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    seen_records = set()
    seen_groups = set()

    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each record must be a dictionary")

        record_id = row.get("record_id")
        group_id = row.get("group_id")

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_id must be a non-empty string")

        if record_id in seen_records:
            raise ValueError("duplicate record_id")

        if group_id in seen_groups:
            raise ValueError("duplicate group_id")

        seen_records.add(record_id)
        seen_groups.add(group_id)

        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")

        pixels = row.get("pixels")
        if not isinstance(pixels, list) or len(pixels) != 4:
            raise ValueError("pixels must contain exactly four rows")

        for pixel_row in pixels:
            if not isinstance(pixel_row, list) or len(pixel_row) != 4:
                raise ValueError("each image row must contain four pixels")

            for pixel in pixel_row:
                if type(pixel) not in (int, float):
                    raise ValueError("pixels must be numeric and not boolean")
                if not math.isfinite(pixel):
                    raise ValueError("pixels must be finite")
                if pixel < 0 or pixel > 255:
                    raise ValueError("pixels must be in the range 0..255")

    return True


def validate_splits(records, splits):
    validate_records(records)

    required = {"train", "validation", "test"}
    if not isinstance(splits, dict) or set(splits) != required:
        raise ValueError("splits must define train, validation, and test")

    known = {row["record_id"]: row["group_id"] for row in records}
    seen_ids = set()
    seen_groups = set()

    for split_name in ("train", "validation", "test"):
        ids = splits[split_name]

        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split_name} split must be a non-empty list")

        for record_id in ids:
            if not isinstance(record_id, str) or record_id not in known:
                raise ValueError("split contains an unknown record ID")

            if record_id in seen_ids:
                raise ValueError("record appears in more than one split")

            group_id = known[record_id]
            if group_id in seen_groups:
                raise ValueError("group appears in more than one split")

            seen_ids.add(record_id)
            seen_groups.add(group_id)

    if seen_ids != set(known):
        raise ValueError("every record must appear in exactly one split")

    return True


class ImagePreprocessor:
    """Fixed 0..255 to 0..1 scaling; fitting is restricted to train IDs."""

    def __init__(self, train_ids):
        if not isinstance(train_ids, list) or not train_ids:
            raise ValueError("train_ids must be a non-empty list")

        if any(not isinstance(item, str) or not item for item in train_ids):
            raise ValueError("train IDs must be non-empty strings")

        if len(set(train_ids)) != len(train_ids):
            raise ValueError("train IDs must be unique")

        self._train_ids = frozenset(train_ids)
        self._fitted = False

    def fit(self, training_records):
        validate_records(training_records)

        supplied_ids = {row["record_id"] for row in training_records}

        if supplied_ids != self._train_ids:
            raise ValueError(
                "fit accepts exactly the declared training records; "
                "validation/test contamination is not allowed"
            )

        self._fitted = True
        return self

    def transform(self, records):
        if not self._fitted:
            raise RuntimeError("call fit with training records before transform")

        validate_records(records)

        images = np.asarray(
            [row["pixels"] for row in records],
            dtype=np.float32
        )

        images = images / np.float32(255.0)
        images = np.expand_dims(images, axis=-1)

        return images


def run_pipeline(fixture):
    records = fixture["records"]
    splits = fixture["splits"]

    validate_records(records)
    validate_splits(records, splits)

    by_id = {row["record_id"]: row for row in records}
    training_records = [by_id[item] for item in splits["train"]]

    preprocessor = ImagePreprocessor(splits["train"])
    preprocessor.fit(training_records)

    transformed = preprocessor.transform(records)

    return {
        "dtype": str(transformed.dtype),
        "shape": list(transformed.shape),
        "first_image": transformed[0].tolist(),
        "diagonal_image": transformed[3].tolist(),
        "minimum": float(transformed.min()),
        "maximum": float(transformed.max()),
        "status": "synthetic pipeline verification only"
    }
