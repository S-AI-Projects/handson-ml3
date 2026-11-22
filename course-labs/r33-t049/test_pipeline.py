"""
Unit tests for Course Lab R33-T049 Input Perturbation Robustness.
"""
import unittest
import json
import os
from pipeline import RobustImagePipeline

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestInputPerturbationRobustness(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)
        self.pipeline = RobustImagePipeline()

    def test_record_order_permutation_invariance(self):
        # Forward order
        preds_forward = self.pipeline.predict_dataset_indexed(self.fixture)

        # Reversed order
        reversed_records = list(reversed(self.fixture))
        preds_reversed = self.pipeline.predict_dataset_indexed(reversed_records)

        # Permuted order
        permuted_records = [self.fixture[2], self.fixture[0], self.fixture[5], self.fixture[1], self.fixture[4], self.fixture[3]]
        preds_permuted = self.pipeline.predict_dataset_indexed(permuted_records)

        self.assertEqual(preds_forward, preds_reversed)
        self.assertEqual(preds_forward, preds_permuted)

    def test_shape_mismatch_rejected(self):
        bad_record = {"record_id": "bad1", "image": [[0]*3 for _ in range(3)]}
        with self.assertRaises(ValueError):
            self.pipeline.predict_sample(bad_record)

    def test_missing_field_rejected(self):
        bad_record = {"record_id": "bad2"}
        with self.assertRaises(KeyError):
            self.pipeline.predict_sample(bad_record)

    def test_expected_predictions(self):
        preds = self.pipeline.predict_dataset_indexed(self.fixture)
        for i in range(1, 4):
            self.assertEqual(preds[f"img{i}"], 0)
        for i in range(4, 7):
            self.assertEqual(preds[f"img{i}"], 1)

if __name__ == "__main__":
    unittest.main()
