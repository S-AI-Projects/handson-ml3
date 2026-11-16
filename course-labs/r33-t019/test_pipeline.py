import copy
import json
import unittest
from pathlib import Path

from pipeline import analyse, classify_image, validate_fixture


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


class TestImageInputRobustness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with FIXTURE_PATH.open(encoding="utf-8") as file:
            cls.data = json.load(file)

    def test_fixture_is_valid(self):
        validate_fixture(self.data)

    def test_fixture_has_six_images(self):
        self.assertEqual(len(self.data["records"]), 6)

    def test_all_images_are_four_by_four(self):
        for record in self.data["records"]:
            self.assertEqual(len(record["pixels"]), 4)
            self.assertTrue(
                all(len(row) == 4 for row in record["pixels"])
            )

    def test_three_black_and_three_diagonal_images(self):
        black_count = 0
        diagonal_count = 0

        for record in self.data["records"]:
            if record["label"] == 0:
                black_count += 1
            elif record["label"] == 1:
                diagonal_count += 1

        self.assertEqual(black_count, 3)
        self.assertEqual(diagonal_count, 3)

    def test_record_ids_are_unique(self):
        ids = [record["record_id"] for record in self.data["records"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_group_ids_are_declared(self):
        for record in self.data["records"]:
            self.assertTrue(record["group_id"])

    def test_all_splits_exist(self):
        splits = {record["split"] for record in self.data["records"]}
        self.assertEqual(splits, {"train", "validation", "test"})

    def test_black_image_is_classified_as_zero(self):
        image = [[0, 0, 0, 0] for _ in range(4)]
        self.assertEqual(classify_image(image), 0)

    def test_white_diagonal_is_classified_as_one(self):
        image = [
            [255, 0, 0, 0],
            [0, 255, 0, 0],
            [0, 0, 255, 0],
            [0, 0, 0, 255]
        ]
        self.assertEqual(classify_image(image), 1)

    def test_all_synthetic_records_are_classified_correctly(self):
        result = analyse(self.data)

        self.assertEqual(result["record_count"], 6)
        self.assertEqual(result["correct_count"], 6)
        self.assertEqual(result["error_count"], 0)

    def test_shuffling_records_preserves_predictions_after_id_alignment(self):
        original = analyse(self.data)

        shuffled_data = copy.deepcopy(self.data)
        shuffled_data["records"].reverse()

        shuffled = analyse(shuffled_data)

        self.assertEqual(
            original["results_by_id"],
            shuffled["results_by_id"]
        )

    def test_missing_field_is_rejected(self):
        data = copy.deepcopy(self.data)
        del data["records"][0]["group_id"]

        with self.assertRaises(ValueError):
            validate_fixture(data)

    def test_wrong_image_shape_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["records"][0]["pixels"] = [[0, 0], [0, 0]]

        with self.assertRaises(ValueError):
            validate_fixture(data)

    def test_unsupported_image_pattern_is_rejected(self):
        image = [
            [255, 255, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        with self.assertRaises(ValueError):
            classify_image(image)

    def test_duplicate_record_id_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["records"][1]["record_id"] = data["records"][0]["record_id"]

        with self.assertRaises(ValueError):
            validate_fixture(data)

    def test_missing_split_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["records"][0]["split"] = "unknown"

        with self.assertRaises(ValueError):
            validate_fixture(data)


if __name__ == "__main__":
    unittest.main()
