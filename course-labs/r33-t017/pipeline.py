"""Reproducible configuration and CPU-only image classification smoke pipeline."""

import json
from pathlib import Path

EXPECTED_CONFIG_KEYS = {
    "schema_version", "seed", "feature_order", "splits", "algorithm"
}
EXPECTED_SPLITS = {"train", "validation", "test"}
EXPECTED_ALGORITHM = {
    "name": "nearest_template",
    "distance": "hamming",
    "tie_break": "lowest_label",
}
EXPECTED_RECORD_KEYS = {"record_id", "group_id", "pixels", "label"}


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    unknown = set(config) - EXPECTED_CONFIG_KEYS
    missing = EXPECTED_CONFIG_KEYS - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")

    if config["schema_version"] != "1.0":
        raise ValueError("Unsupported schema_version.")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer.")
    if config["feature_order"] != ["pixels"]:
        raise ValueError("feature_order must be exactly ['pixels'].")

    splits = config["splits"]
    if not isinstance(splits, dict) or set(splits) != EXPECTED_SPLITS:
        raise ValueError("splits must contain exactly train, validation and test.")

    for name, ids in splits.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Split {name} must be a non-empty list of record IDs.")
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError(f"Split {name} contains an invalid record ID.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Split {name} contains duplicate record IDs.")

    algorithm = config["algorithm"]
    if not isinstance(algorithm, dict):
        raise ValueError("algorithm must be a JSON object.")
    if algorithm != EXPECTED_ALGORITHM:
        raise ValueError(
            "Unsupported or contradictory algorithm settings. "
            f"Expected {EXPECTED_ALGORITHM}."
        )


def validate_fixture(fixture, config):
    if not isinstance(fixture, dict) or not isinstance(fixture.get("records"), list):
        raise ValueError("Fixture must contain a records list.")

    if fixture.get("data_type") != "synthetic_cpu_smoke_fixture":
        raise ValueError("Fixture must be explicitly labelled as synthetic.")

    records = fixture["records"]
    by_id = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Row {index}: record must be an object.")
        if set(record) != EXPECTED_RECORD_KEYS:
            raise ValueError(
                f"Row {index}: fields must be exactly {sorted(EXPECTED_RECORD_KEYS)}."
            )

        record_id = record["record_id"]
        group_id = record["group_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Row {index}: record_id must be a non-empty string.")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{record_id}: group_id must be a non-empty string.")
        if record_id in by_id:
            raise ValueError(f"{record_id}: duplicate record ID.")

        pixels = record["pixels"]
        if not isinstance(pixels, list) or len(pixels) != 4:
            raise ValueError(f"{record_id}: pixels must have exactly 4 rows.")
        for row in pixels:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError(f"{record_id}: every image row must have 4 pixels.")
            for value in row:
                if isinstance(value, bool) or value not in (0, 255):
                    raise ValueError(
                        f"{record_id}: pixels must be integer grayscale values 0 or 255."
                    )

        label = record["label"]
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"{record_id}: label must be integer 0 or 1.")

        by_id[record_id] = record

    split_ids = [
        record_id
        for split in config["splits"].values()
        for record_id in split
    ]
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("A record ID appears in more than one split.")
    if set(split_ids) != set(by_id):
        raise ValueError("Split IDs must cover every fixture record exactly once.")

    group_to_split = {}
    for split_name, ids in config["splits"].items():
        for record_id in ids:
            group_id = by_id[record_id]["group_id"]
            if group_id in group_to_split and group_to_split[group_id] != split_name:
                raise ValueError(f"Group {group_id} appears in multiple splits.")
            group_to_split[group_id] = split_name

    return by_id


def flatten(pixels):
    return [value for row in pixels for value in row]


def train_templates(records, training_ids):
    templates = {}
    for record_id in training_ids:
        record = records[record_id]
        label = record["label"]
        templates.setdefault(label, flatten(record["pixels"]))

    if set(templates) != {0, 1}:
        raise ValueError("Training split must contain examples of both labels 0 and 1.")
    return templates


def predict_one(pixels, templates):
    values = flatten(pixels)
    distances = {
        label: sum(a != b for a, b in zip(values, template))
        for label, template in templates.items()
    }
    return min(distances, key=lambda label: (distances[label], label))


def run_pipeline(fixture, config):
    validate_config(config)
    records = validate_fixture(fixture, config)
    templates = train_templates(records, config["splits"]["train"])

    predictions = {}
    for split_name, ids in config["splits"].items():
        predictions[split_name] = {
            record_id: predict_one(records[record_id]["pixels"], templates)
            for record_id in ids
        }
    return predictions


def run_from_files(fixture_path, config_path):
    fixture = load_json(fixture_path)
    config = load_json(config_path)
    return run_pipeline(fixture, config)
