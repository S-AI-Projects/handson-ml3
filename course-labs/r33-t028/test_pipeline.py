import unittest
import json
import os
import random
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

RobustLinearRegression = pipeline.RobustLinearRegression

class TestInputPerturbationRobustness(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        self.model = RobustLinearRegression()

    def test_fixture_integrity_and_formula(self):
        """Asserts 6 rows, x1=[0..5], x2=1, target=2*x1+1, train r1..r4, val r5, test r6."""
        self.assertEqual(len(self.fixture), 6)
        train_rows = [r for r in self.fixture if r["split"] == "train"]
        val_rows = [r for r in self.fixture if r["split"] == "val"]
        test_rows = [r for r in self.fixture if r["split"] == "test"]
        self.assertEqual(len(train_rows), 4)
        self.assertEqual(len(val_rows), 1)
        self.assertEqual(len(test_rows), 1)
        for r in self.fixture:
            self.assertEqual(r["target"], 2.0 * r["x1"] + 1.0)

    def test_record_order_perturbation_invariance(self):
        """Predictions mapped by record_id are strictly invariant under record order changes."""
        orig_map = self.model.predict_as_id_map(self.fixture)

        # Perturb order with fixed shuffle
        shuffled = list(self.fixture)
        random.Random(42).shuffle(shuffled)
        shuffled_map = self.model.predict_as_id_map(shuffled)

        self.assertEqual(orig_map, shuffled_map)
        for r in self.fixture:
            self.assertEqual(orig_map[r["record_id"]], r["target"])

    def test_missing_field_fails_gracefully(self):
        """Missing required field raises KeyError."""
        perturbed = [{"record_id": "r1", "x2": 1.0}]  # Missing x1
        with self.assertRaises(KeyError):
            self.model.predict(perturbed)

    def test_shape_mismatch_fails_gracefully(self):
        """Malformed non-numeric value raises ValueError."""
        perturbed = [{"record_id": "r1", "x1": "invalid_str", "x2": 1.0}]
        with self.assertRaises(ValueError):
            self.model.predict(perturbed)

    def test_deliberate_mutation_failure(self):
        """Mutating formula prediction causes assertion failure."""
        preds = self.model.predict_as_id_map(self.fixture)
        mutated_target = 999.0
        self.assertNotEqual(preds["r1"], mutated_target)
