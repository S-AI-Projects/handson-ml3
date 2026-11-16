import copy
import json
import unittest
from pathlib import Path

import numpy as np

from pipeline import (
    ImagePreprocessor,
    run_pipeline,
    validate_records,
    validate_splits,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def load_fixture():
    with FIXTURE_PATH.open() as file:
        return json.load(file)


class TestImagePreprocessing(unittest.TestCase):

    def setUp(self):
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.splits = self.fixture["splits"]

    def make_preprocessor(self):
        by_id = {row["record_id"]: row for row in self.records}
        training = [by_id[item] for item in self.splits["train"]]
        processor = ImagePreprocessor(self.splits["train"])
        processor.fit(training)
        return processor

    def test_fixture_contains_six_images(self):
        self.assertEqual(len(self.records), 6)

    def test_fixture_has_three_images_per_class(self):
        labels = [row["label"] for row in self.records]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_record_and_group_ids_are_unique(self):
        record_ids = [row["record_id"] for row in self.records]
        group_ids = [row["group_id"] for row in self.records]
        self.assertEqual(len(record_ids), len(set(record_ids)))
        self.assertEqual(len(group_ids), len(set(group_ids)))

    def test_fixture_schema_is_valid(self):
        self.assertTrue(validate_records(self.records))

    def test_splits_are_disjoint_and_complete(self):
        self.assertTrue(validate_splits(self.records, self.splits))

    def test_train_validation_test_each_have_two_records(self):
        self.assertEqual(len(self.splits["train"]), 2)
        self.assertEqual(len(self.splits["validation"]), 2)
        self.assertEqual(len(self.splits["test"]), 2)

    def test_transform_requires_fit_first(self):
        processor = ImagePreprocessor(self.splits["train"])
        with self.assertRaises(RuntimeError):
            processor.transform(self.records)

    def test_output_dtype_is_float32(self):
        output = self.make_preprocessor().transform(self.records)
        self.assertEqual(output.dtype, np.float32)

    def test_output_shape_has_explicit_channel_axis(self):
        output = self.make_preprocessor().transform(self.records)
        self.assertEqual(output.shape, (6, 4, 4, 1))

    def test_zero_image_scales_to_zero(self):
        output = self.make_preprocessor().transform(self.records)
        np.testing.assert_array_equal(output[0], np.zeros((4, 4, 1), dtype=np.float32))

    def test_white_diagonal_scales_to_one(self):
        output = self.make_preprocessor().transform(self.records)
        expected = np.eye(4, dtype=np.float32).reshape(4, 4, 1)
        np.testing.assert_array_equal(output[3], expected)

    def test_all_output_values_are_in_zero_to_one_range(self):
        output = self.make_preprocessor().transform(self.records)
        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))

    def test_fit_rejects_validation_or_test_contamination(self):
        by_id = {row["record_id"]: row for row in self.records}
        contaminated = [
            by_id[item]
            for item in self.splits["train"] + [self.splits["validation"][0]]
        ]
        processor = ImagePreprocessor(self.splits["train"])

        with self.assertRaises(ValueError):
            processor.fit(contaminated)

    def test_out_of_range_pixel_is_rejected(self):
        bad = copy.deepcopy(self.records)
        bad[0]["pixels"][0][0] = 256
        with self.assertRaises(ValueError):
            validate_records(bad)

    def test_negative_pixel_is_rejected(self):
        bad = copy.deepcopy(self.records)
        bad[0]["pixels"][0][0] = -1
        with self.assertRaises(ValueError):
            validate_records(bad)

    def test_invalid_image_dimensions_are_rejected(self):
        bad = copy.deepcopy(self.records)
        bad[0]["pixels"] = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            validate_records(bad)

    def test_non_numeric_pixel_is_rejected(self):
        bad = copy.deepcopy(self.records)
        bad[0]["pixels"][0][0] = "white"
        with self.assertRaises(ValueError):
            validate_records(bad)

    def test_duplicate_record_id_is_rejected(self):
        bad = copy.deepcopy(self.records)
        bad[1]["record_id"] = bad[0]["record_id"]
        with self.assertRaises(ValueError):
            validate_records(bad)

    def test_split_overlap_is_rejected(self):
        bad_splits = copy.deepcopy(self.splits)
        bad_splits["validation"][0] = bad_splits["train"][0]
        with self.assertRaises(ValueError):
            validate_splits(self.records, bad_splits)

    def test_concrete_pipeline_result(self):
        result = run_pipeline(self.fixture)
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual(result["shape"], [6, 4, 4, 1])
        self.assertEqual(result["minimum"], 0.0)
        self.assertEqual(result["maximum"], 1.0)
        self.assertEqual(
            result["status"],
            "synthetic pipeline verification only"
        )


if __name__ == "__main__":
    unittest.main()
