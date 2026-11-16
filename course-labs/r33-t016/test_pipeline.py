import copy
import json
import unittest
from pathlib import Path

import numpy as np

from pipeline import (
    run_pipeline,
    validate_records,
    validate_splits,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text())


class TestSmokePipeline(unittest.TestCase):

    def setUp(self):
        self.fixture = load_fixture()

    def test_fixture_has_six_records(self):
        self.assertEqual(len(self.fixture["records"]), 6)

    def test_fixture_has_three_records_per_class(self):
        labels = [row["label"] for row in self.fixture["records"]]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_record_ids_are_unique(self):
        ids = [row["record_id"] for row in self.fixture["records"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_group_ids_are_unique(self):
        ids = [row["group_id"] for row in self.fixture["records"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_fixture_schema_is_valid(self):
        self.assertTrue(validate_records(self.fixture["records"]))

    def test_splits_are_valid(self):
        self.assertTrue(
            validate_splits(self.fixture["records"], self.fixture["splits"])
        )

    def test_splits_are_disjoint(self):
        splits = self.fixture["splits"]
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["validation"]) & set(splits["test"]))

    def test_all_records_are_assigned(self):
        all_ids = set()
        for ids in self.fixture["splits"].values():
            all_ids.update(ids)
        self.assertEqual(
            all_ids,
            {row["record_id"] for row in self.fixture["records"]}
        )

    def test_output_has_expected_prediction_shape(self):
        result = run_pipeline(self.fixture)
        self.assertEqual(result["prediction_shape"], [6, 2])

    def test_output_has_six_predictions(self):
        result = run_pipeline(self.fixture)
        self.assertEqual(len(result["predictions"]), 6)

    def test_loss_is_finite(self):
        result = run_pipeline(self.fixture)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(result["loss_is_finite"])

    def test_step_count_is_bounded(self):
        result = run_pipeline(self.fixture)
        self.assertEqual(result["step_count"], 20)
        self.assertLessEqual(result["step_count"], 20)

    def test_sample_count_is_bounded(self):
        result = run_pipeline(self.fixture)
        self.assertLessEqual(result["sample_count"], 32)

    def test_probabilities_are_finite(self):
        result = run_pipeline(self.fixture)
        self.assertTrue(np.all(np.isfinite(result["probabilities"])))

    def test_probabilities_sum_to_one(self):
        result = run_pipeline(self.fixture)
        totals = np.sum(result["probabilities"], axis=1)
        np.testing.assert_allclose(totals, np.ones(6), atol=1e-6)

    def test_run_is_deterministic(self):
        first = run_pipeline(self.fixture)
        second = run_pipeline(self.fixture)
        self.assertEqual(first, second)

    def test_wrong_image_dimensions_are_rejected(self):
        bad = copy.deepcopy(self.fixture)
        bad["records"][0]["pixels"] = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            run_pipeline(bad)

    def test_out_of_range_pixel_is_rejected(self):
        bad = copy.deepcopy(self.fixture)
        bad["records"][0]["pixels"][0][0] = 256
        with self.assertRaises(ValueError):
            run_pipeline(bad)

    def test_negative_pixel_is_rejected(self):
        bad = copy.deepcopy(self.fixture)
        bad["records"][0]["pixels"][0][0] = -1
        with self.assertRaises(ValueError):
            run_pipeline(bad)

    def test_split_overlap_is_rejected(self):
        bad = copy.deepcopy(self.fixture)
        bad["splits"]["validation"][0] = bad["splits"]["train"][0]
        with self.assertRaises(ValueError):
            run_pipeline(bad)

    def test_more_than_twenty_updates_is_rejected(self):
        bad = copy.deepcopy(self.fixture)
        bad["max_updates"] = 21
        with self.assertRaises(ValueError):
            run_pipeline(bad)


if __name__ == "__main__":
    unittest.main()
