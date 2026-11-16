import json
import sys
from pathlib import Path

FIXTURE_PATH = Path(__file__).with_name("fixture.json")
EXPECTED_SPLITS = {"train", "validation", "test"}
EXPECTED_SHAPE = (4, 4)


def validate_record(record, seen_ids):
    if not isinstance(record, dict):
        raise ValueError("Each record must be an object")

    required = {"record_id", "group_id", "pixels", "label", "split"}
    missing = required - set(record)

    if missing:
        raise ValueError("Missing required fields: " + ", ".join(sorted(missing)))

    record_id = record["record_id"]
    group_id = record["group_id"]

    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")

    if record_id in seen_ids:
        raise ValueError("Duplicate record_id: " + record_id)

    seen_ids.add(record_id)

    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")

    if record["split"] not in EXPECTED_SPLITS:
        raise ValueError("Invalid data split")

    if type(record["label"]) is not int or record["label"] not in (0, 1):
        raise ValueError("label must be integer 0 or 1")

    pixels = record["pixels"]

    if not isinstance(pixels, list) or len(pixels) != EXPECTED_SHAPE[0]:
        raise ValueError("Image must have exactly 4 rows")

    for row in pixels:
        if not isinstance(row, list) or len(row) != EXPECTED_SHAPE[1]:
            raise ValueError("Every image row must contain exactly 4 pixels")

        for value in row:
            if type(value) is not int or value not in (0, 255):
                raise ValueError("Pixels must be integer 0 or 255")


def validate_fixture(data):
    if data.get("dataset_type") != "synthetic_simulated":
        raise ValueError("Fixture must be labelled synthetic_simulated")

    if data.get("image_shape") != [4, 4]:
        raise ValueError("Expected image shape [4, 4]")

    if data.get("seed") != 42:
        raise ValueError("Expected fixed seed 42")

    records = data.get("records")

    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Fixture must contain exactly six records")

    seen_ids = set()
    seen_splits = set()

    for record in records:
        validate_record(record, seen_ids)
        seen_splits.add(record["split"])

    if seen_splits != EXPECTED_SPLITS:
        raise ValueError("Train, validation and test splits are required")


def classify_image(pixels):
    flat = [value for row in pixels for value in row]

    if all(value == 0 for value in flat):
        return 0

    diagonal = [
        [255, 0, 0, 0],
        [0, 255, 0, 0],
        [0, 0, 255, 0],
        [0, 0, 0, 255]
    ]

    if pixels == diagonal:
        return 1

    raise ValueError("Unsupported image pattern")


def analyse(data):
    validate_fixture(data)

    results = []

    for record in data["records"]:
        prediction = classify_image(record["pixels"])

        results.append({
            "record_id": record["record_id"],
            "group_id": record["group_id"],
            "split": record["split"],
            "expected_label": record["label"],
            "predicted_label": prediction,
            "correct": prediction == record["label"]
        })

    results_by_id = {
        item["record_id"]: item for item in results
    }

    return {
        "dataset_type": data["dataset_type"],
        "verification_type": "synthetic_pipeline_verification",
        "record_count": len(results),
        "correct_count": sum(item["correct"] for item in results),
        "error_count": sum(not item["correct"] for item in results),
        "results": results,
        "results_by_id": results_by_id
    }


def main():
    fixture_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE_PATH

    with fixture_path.open(encoding="utf-8") as file:
        data = json.load(file)

    print(json.dumps(analyse(data), indent=2))


if __name__ == "__main__":
    main()
