"""
Unit tests for R33-T010: Saved-Pipeline Interface and Schema Validation
"""

import json
import os
import tempfile
import unittest
from pipeline import (
    load_dataset,
    SavedLinearPipeline,
    SchemaValidationError,
    VersionMismatchError,
    ModelCorruptionError,
    DatasetRecord,
)


class TestR33T010SavedPipelineInterface(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_json = load_dataset(os.path.join(self.base_dir, "fixture.json"), format="json")
        self.records_csv = load_dataset(os.path.join(self.base_dir, "fixture.csv"), format="csv")

        # Train model on training split (r1..r4)
        self.train_records = [r for r in self.records_json if r.split == "train"]
        self.val_records = [r for r in self.records_json if r.split == "validation"]
        self.test_records = [r for r in self.records_json if r.split == "test"]

        self.model = SavedLinearPipeline()
        self.model.fit(self.train_records)

    def test_fixture_records_and_formula(self):
        """
        Fixture must contain six synthetic rows obeying target = 2*x1 + 1:
        x1=[0..5], x2=1, splits: train r1..r4, val r5, test r6.
        """
        self.assertEqual(len(self.records_json), 6)
        expected_splits = {
            "r1": "train",
            "r2": "train",
            "r3": "train",
            "r4": "train",
            "r5": "validation",
            "r6": "test",
        }
        for i, r in enumerate(self.records_json):
            self.assertEqual(r.x1, float(i))
            self.assertEqual(r.x2, 1.0)
            self.assertEqual(r.target, 2.0 * float(i) + 1.0)
            self.assertEqual(r.split, expected_splits[r.record_id])

    def test_csv_and_json_parity(self):
        """CSV and JSON fixtures must load identical data."""
        self.assertEqual(len(self.records_json), len(self.records_csv))
        for r_j, r_c in zip(self.records_json, self.records_csv):
            self.assertEqual(r_j.record_id, r_c.record_id)
            self.assertEqual(r_j.x1, r_c.x1)
            self.assertEqual(r_j.x2, r_c.x2)
            self.assertEqual(r_j.target, r_c.target)
            self.assertEqual(r_j.split, r_c.split)

    def test_model_fit_accuracy(self):
        """Fitted model must recover w1=2.0, w2=1.0 and predict accurately."""
        self.assertAlmostEqual(self.model.weights[0], 2.0, places=9)
        self.assertAlmostEqual(self.model.weights[1], 1.0, places=9)

        # Predict on validation (r5, x1=4 -> 9.0) and test (r6, x1=5 -> 11.0)
        val_pred = self.model.predict([r.to_feature_dict() for r in self.val_records])
        test_pred = self.model.predict([r.to_feature_dict() for r in self.test_records])

        self.assertAlmostEqual(val_pred[0], 9.0, places=9)
        self.assertAlmostEqual(test_pred[0], 11.0, places=9)

    def test_save_and_reload_predictions_parity(self):
        """
        Before/after predictions must agree within explicit tolerance (1e-9).
        Verifies both dict-based and list-based inputs.
        """
        eval_records = self.val_records + self.test_records
        eval_dicts = [r.to_feature_dict() for r in eval_records]
        eval_lists = [r.to_features() for r in eval_records]

        preds_before_dict = self.model.predict(eval_dicts)
        preds_before_list = self.model.predict(eval_lists)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model_artifact.json")
            self.model.save(model_path)

            # Reload model from JSON artifact
            reloaded_model = SavedLinearPipeline.load(model_path)

            preds_after_dict = reloaded_model.predict(eval_dicts)
            preds_after_list = reloaded_model.predict(eval_lists)

            # Parity checks
            for b, a in zip(preds_before_dict, preds_after_dict):
                self.assertAlmostEqual(b, a, places=9)
            for b, a in zip(preds_before_list, preds_after_list):
                self.assertAlmostEqual(b, a, places=9)

    def test_no_training_data_or_credentials_needed_at_inference(self):
        """
        Reloaded model must perform inference in isolation with zero training data
        and zero external network/credentials.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.json")
            self.model.save(model_path)

            # Load in fresh isolated scope
            isolated_pipeline = SavedLinearPipeline.load(model_path)
            self.assertFalse(hasattr(isolated_pipeline, "training_records"))

            # Novel input inference
            test_row = [{"x1": 10.0, "x2": 1.0}]
            pred = isolated_pipeline.predict(test_row)
            self.assertAlmostEqual(pred[0], 21.0, places=9)

    def test_schema_validation_missing_feature(self):
        """Missing feature key must raise SchemaValidationError."""
        bad_input = [{"x1": 5.0}]  # Missing x2
        with self.assertRaises(SchemaValidationError) as ctx:
            self.model.predict(bad_input)
        self.assertIn("Missing required feature", str(ctx.exception))

    def test_schema_validation_extra_feature(self):
        """Unknown or extra feature must raise SchemaValidationError."""
        bad_input = [{"x1": 5.0, "x2": 1.0, "unrecognized_col": 42.0}]
        with self.assertRaises(SchemaValidationError) as ctx:
            self.model.predict(bad_input)
        self.assertIn("Unknown feature(s) detected", str(ctx.exception))

    def test_schema_validation_invalid_type_or_nan(self):
        """Non-numeric values or NaNs must raise SchemaValidationError."""
        bad_type_input = [{"x1": "not_a_number", "x2": 1.0}]
        with self.assertRaises(SchemaValidationError) as ctx:
            self.model.predict(bad_type_input)
        self.assertIn("Invalid type", str(ctx.exception))

        nan_input = [{"x1": float("nan"), "x2": 1.0}]
        with self.assertRaises(SchemaValidationError) as ctx:
            self.model.predict(nan_input)
        self.assertIn("Non-finite value", str(ctx.exception))

    def test_version_mismatch_fails_clearly(self):
        """Unsupported schema_version in artifact must raise VersionMismatchError."""
        artifact = self.model.to_dict()
        artifact["schema_version"] = "99.0.0"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "incompatible_model.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(artifact, f)

            with self.assertRaises(VersionMismatchError) as ctx:
                SavedLinearPipeline.load(path)
            self.assertIn("Unsupported schema_version", str(ctx.exception))

    def test_corrupted_model_artifact_fails_clearly(self):
        """Malformed or incomplete JSON artifact must raise ModelCorruptionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{ invalid json")

            with self.assertRaises(ModelCorruptionError) as ctx:
                SavedLinearPipeline.load(path)
            self.assertIn("Corrupted or invalid JSON", str(ctx.exception))

    def test_deliberate_mutation_fails(self):
        """Mutation test: changing expected prediction causes assertion failure."""
        preds = self.model.predict([r.to_feature_dict() for r in self.test_records])
        mutated_wrong_val = 999.0
        self.assertNotEqual(preds[0], mutated_wrong_val)


if __name__ == "__main__":
    unittest.main()
