"""
Unit tests for Course Lab R33-T043 minimal majority-class baseline.
"""
import unittest
import json
import os
from pipeline import MajorityClassBaseline

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestMinimalBaseline(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)

    def test_fixture_integrity(self):
        self.assertEqual(len(self.fixture), 6)
        labels = [r["label"] for r in self.fixture]
        self.assertEqual(labels, [0, 0, 0, 1, 1, 1])
        for r in self.fixture:
            self.assertEqual(len(r["image"]), 4)
            self.assertEqual(len(r["image"][0]), 4)

    def test_majority_baseline_balanced_fit(self):
        baseline = MajorityClassBaseline()
        # Even split [0, 0, 0, 1, 1, 1] -> tie breaker prefers class 0
        baseline.fit(self.fixture)
        self.assertEqual(baseline.majority_class, 0)
        preds = baseline.predict(self.fixture)
        self.assertEqual(preds, [0, 0, 0, 0, 0, 0])
        eval_metrics = baseline.evaluate(self.fixture)
        self.assertEqual(eval_metrics["accuracy"], 0.5)

    def test_majority_baseline_imbalanced_fit(self):
        # 3 class 1 and 1 class 0
        imbalanced = [self.fixture[0], self.fixture[3], self.fixture[4], self.fixture[5]]
        baseline = MajorityClassBaseline().fit(imbalanced)
        self.assertEqual(baseline.majority_class, 1)
        preds = baseline.predict(self.fixture)
        self.assertEqual(preds, [1, 1, 1, 1, 1, 1])

    def test_unfitted_predict_raises_error(self):
        baseline = MajorityClassBaseline()
        with self.assertRaises(ValueError):
            baseline.predict(self.fixture)

    def test_empty_fit_raises_error(self):
        baseline = MajorityClassBaseline()
        with self.assertRaises(ValueError):
            baseline.fit([])

if __name__ == "__main__":
    unittest.main()
