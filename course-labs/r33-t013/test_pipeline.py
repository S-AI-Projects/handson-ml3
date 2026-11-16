import copy
import json
import unittest
from pathlib import Path

from pipeline import (
    validate_records,
    validate_splits,
    fit_majority_baseline,
    run_experiment,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def load_fixture():
    with FIXTURE_PATH.open() as file:
        return json.load(file)


class TestImageBaseline(unittest.TestCase):

    def setUp(self):
        self.data = load_fixture()

    def test_fixture_has_six_records(self):
        self.assertEqual(len(self.data["records"]), 6)

    def test_fixture_contains_three_of_each_label(self):
        labels = [row["label"] for row in self.data["records"]]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_all_images_are_four_by_four(self):
        for row in self.data["records"]:
            self.assertEqual(len(row["pixels"]), 4)
            self.assertTrue(all(len(line) == 4 for line in row["pixels"]))

    def test_valid_fixture_passes(self):
        self.assertEqual(validate_records(self.data["records"]), [])

    def test_record_ids_are_unique(self):
        ids = [row["record_id"] for row in self.data["records"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_splits_are_disjoint(self):
        self.assertEqual(
            validate_splits(self.data["records"], self.data["splits"]), []
        )

    def test_baseline_uses_training_labels(self):
        label = fit_majority_baseline(
            self.data["records"], self.data["splits"]["train"]
        )
        self.assertEqual(label, 0)

    def test_concrete_baseline_result(self):
        result = run_experiment(self.data)
        self.assertEqual(result["baseline"], "majority_class")
        self.assertEqual(result["training_majority_label"], 0)
        self.assertEqual(result["test_predictions"], {"r3": 0})
        self.assertEqual(result["test_accuracy"], 1.0)

    def test_missing_record_id_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        del records[0]["record_id"]
        errors = validate_records(records)
        self.assertTrue(any(e["field"] == "record_id" for e in errors))

    def test_duplicate_record_id_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[1]["record_id"] = records[0]["record_id"]
        errors = validate_records(records)
        self.assertTrue(any("duplicate" in e["reason"] for e in errors))

    def test_malformed_image_shape_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[0]["pixels"] = [[0, 0], [0, 0]]
        errors = validate_records(records)
        self.assertTrue(any(e["field"] == "pixels" for e in errors))

    def test_non_numeric_pixel_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[0]["pixels"][0][0] = "white"
        errors = validate_records(records)
        self.assertTrue(any(e["field"] == "pixels" for e in errors))

    def test_invalid_label_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[0]["label"] = 2
        errors = validate_records(records)
        self.assertTrue(any(e["field"] == "label" for e in errors))

    def test_group_leakage_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[5]["group_id"] = records[0]["group_id"]
        errors = validate_splits(records, self.data["splits"])
        self.assertTrue(any("group" in e["field"] for e in errors))

    def test_record_overlap_is_rejected(self):
        splits = copy.deepcopy(self.data["splits"])
        splits["test"].append(splits["train"][0])
        errors = validate_splits(self.data["records"], splits)
        self.assertTrue(any("more than one split" in e["reason"] for e in errors))

    def test_empty_training_split_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_majority_baseline(self.data["records"], [])


if __name__ == "__main__":
    unittest.main()
