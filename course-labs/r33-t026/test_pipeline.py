import unittest
import json
import os
import math
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

BoundedCPUSmokeClassifier = pipeline.BoundedCPUSmokeClassifier

class TestCPUSmokeRun(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)

    def test_fixture_integrity(self):
        """Asserts 8 synthetic rows x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.fixture), 8)
        self.assertEqual([r["x1"] for r in self.fixture], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual([r["x2"] for r in self.fixture], [1.0]*8)
        self.assertEqual([r["label"] for r in self.fixture], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_bounded_training_smoke_run(self):
        """Run has finite loss, recorded step count <= 20, and bounded sample count."""
        clf = BoundedCPUSmokeClassifier(seed=42, max_steps=15)
        res = clf.train_smoke_run(self.fixture)

        self.assertTrue(res["is_finite_loss"])
        self.assertTrue(math.isfinite(res["final_loss"]))
        self.assertLessEqual(res["recorded_steps"], 20)
        self.assertEqual(res["recorded_steps"], 15)
        self.assertEqual(res["model_type"], "deterministic_bounded_cpu_smoke_model")

    def test_prediction_shape_and_interface(self):
        """Predictions match input sample count exactly."""
        clf = BoundedCPUSmokeClassifier(seed=42, max_steps=10)
        clf.train_smoke_run(self.fixture)
        preds = clf.predict(self.fixture)
        self.assertEqual(len(preds), len(self.fixture))
        for p in preds:
            self.assertIn(p, (0, 1))

    def test_sample_limit_enforced(self):
        """More than 32 samples raises ValueError."""
        clf = BoundedCPUSmokeClassifier()
        large_fixture = self.fixture * 5  # 40 rows
        with self.assertRaises(ValueError):
            clf.train_smoke_run(large_fixture)

    def test_deliberate_mutation_failure(self):
        """Mutating recorded step count fails assertion."""
        clf = BoundedCPUSmokeClassifier(seed=42, max_steps=15)
        res = clf.train_smoke_run(self.fixture)
        mutated_steps = 999
        self.assertNotEqual(res["recorded_steps"], mutated_steps)
