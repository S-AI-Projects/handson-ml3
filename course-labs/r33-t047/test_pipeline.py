"""
Unit tests for Course Lab R33-T047 Safe Pickle-Free Model Persistence.
"""
import unittest
import json
import os
import tempfile
from pipeline import SafePretrainedPipeline

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestSafeModelPersistence(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)
        self.model = SafePretrainedPipeline()

    def test_save_and_reload_exact_agreement(self):
        initial_preds = self.model.predict_dataset(self.fixture)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.model.save_json(tmp_path)
            loaded_model = SafePretrainedPipeline.load_json(tmp_path)
            reloaded_preds = loaded_model.predict_dataset(self.fixture)
            self.assertEqual(initial_preds, reloaded_preds)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_reject_pickle_extension(self):
        with self.assertRaises(ValueError):
            self.model.save_json("model.pkl")
        with self.assertRaises(ValueError):
            SafePretrainedPipeline.load_json("model.pickle")

    def test_reject_schema_version_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            bad_data = {"schema_version": "9.9.9", "weights": [0.0]*16, "bias": 0.0, "threshold": 0.5}
            with open(tmp_path, "w") as f:
                json.dump(bad_data, f)
            with self.assertRaises(ValueError) as ctx:
                SafePretrainedPipeline.load_json(tmp_path)
            self.assertIn("Incompatible schema version", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_input_shape_validation_at_inference(self):
        # 3x3 image must raise ValueError
        bad_sample = [[0.0]*3 for _ in range(3)]
        with self.assertRaises(ValueError):
            self.model.predict_sample(bad_sample)

    def test_no_credentials_needed_at_inference(self):
        # Model predicts offline without network access
        preds = self.model.predict_dataset(self.fixture)
        self.assertEqual(len(preds), 6)

if __name__ == "__main__":
    unittest.main()
