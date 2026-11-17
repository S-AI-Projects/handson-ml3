"""
Unit tests for R33-T007: Reproducible Configuration and Parameter Schema
"""

import json
import os
import unittest
from pipeline import (
    load_dataset,
    run_pipeline,
    ExperimentConfig,
    AlgorithmSettings,
    ConfigValidationError,
    SchemaError,
    DeterministicRegressor,
)


class TestR33T007ConfigReproducibility(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_json = load_dataset(os.path.join(self.base_dir, "fixture.json"), format="json")
        self.records_csv = load_dataset(os.path.join(self.base_dir, "fixture.csv"), format="csv")
        with open(os.path.join(self.base_dir, "config.json"), "r", encoding="utf-8") as f:
            self.raw_config_dict = json.load(f)
        self.config = ExperimentConfig.from_dict(self.raw_config_dict)

    def test_fixture_records_and_formula_dimensions(self):
        """
        Fixture must contain six synthetic rows: x1=[0,1,2,3,4,5], x2=1, target=2*x1+1.
        Splits: train r1..r4, validation r5, test r6.
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

    def test_reloaded_configuration_reconstructs_identical_splits_and_predictions(self):
        """
        Reloaded configuration must reconstruct the exact same split and predictions.
        Weights must match formula (w_x1=2.0, w_x2=1.0).
        """
        # Run original
        result_orig = run_pipeline(self.records_json, self.config)

        # Serialize to JSON and reload
        config_json = self.config.to_json()
        reloaded_config = ExperimentConfig.from_json(config_json)

        # Run reloaded
        result_reloaded = run_pipeline(self.records_json, reloaded_config)

        # Assert identical split assignments
        self.assertEqual(result_orig.train_record_ids, result_reloaded.train_record_ids)
        self.assertEqual(result_orig.val_record_ids, result_reloaded.val_record_ids)
        self.assertEqual(result_orig.test_record_ids, result_reloaded.test_record_ids)

        # Assert identical predictions across all splits
        for rid, val in result_orig.train_predictions.items():
            self.assertAlmostEqual(val, result_reloaded.train_predictions[rid], places=9)
        for rid, val in result_orig.val_predictions.items():
            self.assertAlmostEqual(val, result_reloaded.val_predictions[rid], places=9)
        for rid, val in result_orig.test_predictions.items():
            self.assertAlmostEqual(val, result_reloaded.test_predictions[rid], places=9)

        # Verify exact target predictions
        self.assertAlmostEqual(result_orig.weights["x1"], 2.0, places=9)
        self.assertAlmostEqual(result_orig.weights["x2"], 1.0, places=9)
        self.assertAlmostEqual(result_orig.val_predictions["r5"], 9.0, places=9)
        self.assertAlmostEqual(result_orig.test_predictions["r6"], 11.0, places=9)

        # Zero MSE on deterministic formula
        self.assertAlmostEqual(result_orig.train_mse, 0.0, places=9)
        self.assertAlmostEqual(result_orig.val_mse, 0.0, places=9)
        self.assertAlmostEqual(result_orig.test_mse, 0.0, places=9)

    def test_unknown_top_level_field_rejected(self):
        """Reject configuration with unknown top-level field."""
        bad_dict = dict(self.raw_config_dict)
        bad_dict["extra_unrecognized_key"] = "forbidden_value"
        with self.assertRaises(ConfigValidationError) as ctx:
            ExperimentConfig.from_dict(bad_dict)
        self.assertIn("Unknown configuration field(s) detected", str(ctx.exception))

    def test_unknown_algorithm_setting_rejected(self):
        """Reject configuration with unknown algorithm setting."""
        bad_dict = json.loads(json.dumps(self.raw_config_dict))
        bad_dict["algorithm_settings"]["learning_rate"] = 0.01  # Not an allowed key for LinearRegression
        with self.assertRaises(ConfigValidationError) as ctx:
            ExperimentConfig.from_dict(bad_dict)
        self.assertIn("Unknown algorithm setting(s) detected", str(ctx.exception))

    def test_contradictory_overlapping_splits_rejected(self):
        """Reject configuration where a record ID appears in multiple splits."""
        bad_dict = json.loads(json.dumps(self.raw_config_dict))
        bad_dict["split_ids"]["validation"].append("r1")  # r1 in train and validation
        with self.assertRaises(ConfigValidationError) as ctx:
            ExperimentConfig.from_dict(bad_dict)
        self.assertIn("Contradictory split IDs", str(ctx.exception))

    def test_contradictory_algorithm_settings_rejected(self):
        """Reject contradictory algorithm parameters (e.g. negative alpha, zero max_iter)."""
        bad_dict = json.loads(json.dumps(self.raw_config_dict))
        bad_dict["algorithm_settings"]["regularization_alpha"] = -1.0
        with self.assertRaises(ConfigValidationError) as ctx:
            ExperimentConfig.from_dict(bad_dict)
        self.assertIn("regularization_alpha cannot be negative", str(ctx.exception))

        bad_dict2 = json.loads(json.dumps(self.raw_config_dict))
        bad_dict2["algorithm_settings"]["max_iter"] = 0
        with self.assertRaises(ConfigValidationError) as ctx:
            ExperimentConfig.from_dict(bad_dict2)
        self.assertIn("max_iter must be positive", str(ctx.exception))

    def test_incompatible_input_shape_rejected(self):
        """Feeding incompatible feature dimensions must raise SchemaError."""
        regressor = DeterministicRegressor(regularization_alpha=0.0)
        X_train = [[0.0, 1.0], [1.0, 1.0]]
        y_train = [1.0, 3.0]
        regressor.fit(X_train, y_train, feature_names=["x1", "x2"])

        # Incompatible shape: 3 features instead of 2
        bad_X = [[1.0, 2.0, 3.0]]
        with self.assertRaises(SchemaError) as ctx:
            regressor.predict(bad_X)
        self.assertIn("Incompatible input shape", str(ctx.exception))

    def test_unknown_split_id_rejected(self):
        """Split referencing record ID not present in dataset must be rejected."""
        bad_dict = json.loads(json.dumps(self.raw_config_dict))
        bad_dict["split_ids"]["test"] = ["r99"]
        available_ids = {r.record_id for r in self.records_json}
        with self.assertRaises(ConfigValidationError) as ctx:
            ExperimentConfig.from_dict(bad_dict, available_record_ids=available_ids)
        self.assertIn("Split IDs not found in dataset", str(ctx.exception))

    def test_deliberate_mutation_fails(self):
        """Mutation test: changing expected prediction value triggers assertion failure."""
        result = run_pipeline(self.records_json, self.config)
        mutated_expected = 123.456
        self.assertNotEqual(result.test_predictions["r6"], mutated_expected)


if __name__ == "__main__":
    unittest.main()
