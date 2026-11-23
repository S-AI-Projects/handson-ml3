import unittest
import json
import os
import tempfile
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

SafeSavedClassificationPipeline = pipeline.SafeSavedClassificationPipeline

class TestSavedPipelineInterface(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        # Decision boundary at x1=3.5 (weights=[1.0, 0.0], bias=-3.5, threshold=0.0)
        self.model = SafeSavedClassificationPipeline(weights=[1.0, 0.0], bias=-3.5, threshold=0.0)

    def test_fixture_integrity(self):
        """Asserts 8 synthetic rows x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.fixture), 8)
        self.assertEqual([r["x1"] for r in self.fixture], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual([r["label"] for r in self.fixture], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_serialization_and_reloaded_predictions(self):
        """Reloaded pipeline reproduces exact predictions as original."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.model.save_json(tmp_path)
            reloaded = SafeSavedClassificationPipeline.load_json(tmp_path)

            orig_preds = self.model.predict(self.fixture)
            reloaded_preds = reloaded.predict(self.fixture)
            self.assertEqual(orig_preds, reloaded_preds)
            self.assertEqual(orig_preds, [0, 0, 0, 0, 1, 1, 1, 1])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_untrusted_pickle_rejected(self):
        """Loading non-JSON / pickle paths raises ValueError."""
        with self.assertRaises(ValueError):
            SafeSavedClassificationPipeline.load_json("model.pkl")

    def test_schema_validator_rejects_missing_field(self):
        """Missing field in record raises ValueError."""
        malformed = [{"record_id": "r1", "x1": 1.0, "x2": 1.0}]  # missing group_id
        with self.assertRaises(ValueError):
            self.model.predict(malformed)

    def test_schema_validator_rejects_malformed_types(self):
        """Non-numeric feature values raise ValueError."""
        malformed = [{"record_id": "r1", "group_id": "g1", "x1": "string_bad", "x2": 1.0}]
        with self.assertRaises(ValueError):
            self.model.predict(malformed)

    def test_deliberate_mutation_failure(self):
        """Mutating predicted output causes assertion failure."""
        preds = self.model.predict(self.fixture)
        mutated_preds = [1]*8
        self.assertNotEqual(preds, mutated_preds)
