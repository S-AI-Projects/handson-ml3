"""
Unit tests for Course Lab R33-T050 Bounded CPU Image Pipeline Smoke Test.
"""
import unittest
import json
import os
import math
from pipeline import TinyImagePipeline

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestBoundedImagePipelineSmoke(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)
        self.pipeline = TinyImagePipeline(seed=42)

    def test_smoke_epoch_bounded_and_finite_loss(self):
        res = self.pipeline.train_smoke_epoch(self.fixture, max_steps=20)
        self.assertTrue(res["is_finite_loss"])
        self.assertFalse(math.isnan(res["avg_loss"]))
        self.assertFalse(math.isinf(res["avg_loss"]))
        self.assertLessEqual(res["steps_executed"], 20)
        self.assertEqual(res["steps_executed"], 6)

    def test_prediction_shape(self):
        self.pipeline.train_smoke_epoch(self.fixture)
        preds = self.pipeline.predict(self.fixture)
        self.assertEqual(len(preds), len(self.fixture))
        for p in preds:
            self.assertIn(p, (0, 1))

    def test_deterministic_reproducibility(self):
        p1 = TinyImagePipeline(seed=42)
        r1 = p1.train_smoke_epoch(self.fixture)
        p2 = TinyImagePipeline(seed=42)
        r2 = p2.train_smoke_epoch(self.fixture)
        self.assertEqual(r1["avg_loss"], r2["avg_loss"])
        self.assertEqual(p1.weights, p2.weights)

    def test_bounds_exceeded_error(self):
        oversized = self.fixture * 6  # 36 samples > 32 max
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.train_smoke_epoch(oversized)
        self.assertIn("bounds exceeded", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
