import unittest
import json
import os
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

RetainedScaler = pipeline.RetainedScaler
NearestCentroidClassifier = pipeline.NearestCentroidClassifier
split_manifest = pipeline.split_manifest

class TestPreprocessingContract(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)

    def test_fixture_integrity_and_shape(self):
        """Assert fixture has 6 points with valid schema (record_id, x1, x2)."""
        self.assertEqual(len(self.fixture), 6)
        for r in self.fixture:
            self.assertIn("record_id", r)
            self.assertIsInstance(r["x1"], (int, float))
            self.assertIsInstance(r["x2"], (int, float))

    def test_nearest_centroid_assignments(self):
        """For fixed centroids (0,0) and (5,5), nearest assignments are 0,0,0,1,1,1."""
        clf = NearestCentroidClassifier()
        assignments = clf.assign(self.fixture)
        self.assertEqual(assignments, [0, 0, 0, 1, 1, 1])

    def test_retained_scaler_train_separation(self):
        """Parameters are fit on train only, and held-out data does not alter fitted params."""
        splits = split_manifest(self.fixture)
        scaler = RetainedScaler()
        scaler.fit(splits["train"])
        mean_before = list(scaler.mean)
        std_before = list(scaler.std)

        transformed_test = scaler.transform(splits["test"])
        self.assertEqual(len(transformed_test), len(splits["test"]))
        # Parameters must remain strictly unchanged
        self.assertEqual(scaler.mean, mean_before)
        self.assertEqual(scaler.std, std_before)

    def test_transform_before_fit_rejected(self):
        """Transforming before fit must raise RuntimeError."""
        scaler = RetainedScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(self.fixture)

    def test_contamination_fixture_rejected(self):
        """Deliberate contamination fixture (invalid schema or types) must be rejected."""
        scaler = RetainedScaler()
        scaler.fit(self.fixture)
        contaminated = [{"record_id": "bad1", "x1": "not-a-number", "x2": 0.0}]
        with self.assertRaises(ValueError):
            scaler.transform(contaminated)

    def test_deliberate_mutation_failure(self):
        """Check that test fails if expected assignment is mutated."""
        clf = NearestCentroidClassifier()
        assignments = clf.assign(self.fixture)
        mutated_expectation = [1, 1, 1, 0, 0, 0]
        self.assertNotEqual(assignments, mutated_expectation)
