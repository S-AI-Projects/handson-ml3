"""
Unit tests for R33-T037: Record reproducible configuration
===========================================================
Covers:
  - Configuration serialization and exact deserialization
  - Prediction reproducibility on reloaded configuration
  - Rejection of unknown top-level and algorithm fields
  - Rejection of contradictory configurations (split overlap, invalid bounds)
  - Rejection of incompatible input shapes / missing features
  - Exact fixture sequence verification ([10,12,11,13,15,14,16,18,17,19], d1..d10)
  - Train (6), Validation (2), Test (2) split verification
  - Deliberate wrong expectation detection

Standard library only. Offline CPU execution.
"""

import copy
import json
import sys
import unittest

sys.path.insert(0, "course-labs/r33-t037")
from pipeline import (
    EXPECTED_TEST_IDS,
    EXPECTED_TRAIN_IDS,
    EXPECTED_VAL_IDS,
    SYNTHETIC_FIXTURE,
    AutoregressiveSequenceModel,
    PipelineConfig,
    run_pipeline,
)


class TestConfigurationSerializationAndReproducibility(unittest.TestCase):

    def test_fixture_integrity(self):
        """Verify fixture values [10,12,11,13,15,14,16,18,17,19] and IDs d1..d10."""
        expected_values = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0]
        expected_ids = [f"d{i}" for i in range(1, 11)]

        self.assertEqual(len(SYNTHETIC_FIXTURE), 10)
        self.assertEqual([r["record_id"] for r in SYNTHETIC_FIXTURE], expected_ids)
        self.assertEqual([r["value"] for r in SYNTHETIC_FIXTURE], expected_values)

    def test_split_sizes_and_ordering(self):
        """First 6 train, next 2 validation, final 2 test; no shuffling."""
        config = PipelineConfig()
        self.assertEqual(config.split_ids["train"], EXPECTED_TRAIN_IDS)
        self.assertEqual(config.split_ids["validation"], EXPECTED_VAL_IDS)
        self.assertEqual(config.split_ids["test"], EXPECTED_TEST_IDS)
        self.assertEqual(len(config.split_ids["train"]), 6)
        self.assertEqual(len(config.split_ids["validation"]), 2)
        self.assertEqual(len(config.split_ids["test"]), 2)

    def test_reloaded_config_reconstructs_identical_predictions(self):
        """Reloaded JSON configuration produces bit-for-bit identical predictions."""
        config_original = PipelineConfig(seed=42)
        json_repr = config_original.to_json()

        config_reloaded = PipelineConfig.from_json(json_repr)

        res_orig = run_pipeline(config_original)
        res_reloaded = run_pipeline(config_reloaded)

        self.assertEqual(res_orig["train_ids"], res_reloaded["train_ids"])
        self.assertEqual(res_orig["val_ids"], res_reloaded["val_ids"])
        self.assertEqual(res_orig["test_ids"], res_reloaded["test_ids"])
        self.assertEqual(res_orig["val_predictions"], res_reloaded["val_predictions"])
        self.assertEqual(res_orig["test_predictions"], res_reloaded["test_predictions"])
        self.assertEqual(res_orig["fitted_weight"], res_reloaded["fitted_weight"])
        self.assertEqual(res_orig["fitted_bias"], res_reloaded["fitted_bias"])


class TestConfigurationRejectionOfUnknownAndContradictorySettings(unittest.TestCase):

    def test_reject_unknown_top_level_field(self):
        """Unknown top-level field in dict must raise ValueError."""
        data = PipelineConfig().to_dict()
        data["unexpected_custom_field"] = "malicious_or_unknown"
        with self.assertRaises(ValueError) as ctx:
            PipelineConfig.from_dict(data)
        self.assertIn("Unknown top-level configuration keys", str(ctx.exception))

    def test_reject_unknown_algorithm_field(self):
        """Unknown field inside algorithm_settings must raise ValueError."""
        data = PipelineConfig().to_dict()
        data["algorithm_settings"]["bogus_learning_rate_decay"] = 0.99
        with self.assertRaises(ValueError) as ctx:
            PipelineConfig.from_dict(data)
        self.assertIn("Unknown algorithm_settings keys", str(ctx.exception))

    def test_reject_contradictory_split_overlap(self):
        """Overlapping split IDs (train and val) must raise ValueError."""
        data = PipelineConfig().to_dict()
        # Add d6 (train) into validation
        data["split_ids"]["validation"].append("d6")
        with self.assertRaises(ValueError) as ctx:
            PipelineConfig.from_dict(data)
        self.assertIn("Overlap found", str(ctx.exception))

    def test_reject_contradictory_clip_bounds(self):
        """clip_min >= clip_max must raise ValueError."""
        data = PipelineConfig().to_dict()
        data["algorithm_settings"]["clip_min"] = 100.0
        data["algorithm_settings"]["clip_max"] = 50.0
        with self.assertRaises(ValueError) as ctx:
            PipelineConfig.from_dict(data)
        self.assertIn("clip_min must be strictly less than clip_max", str(ctx.exception))

    def test_reject_invalid_schema_version(self):
        """Unrecognized schema version must raise ValueError."""
        data = PipelineConfig().to_dict()
        data["schema_version"] = "9.9.9"
        with self.assertRaises(ValueError) as ctx:
            PipelineConfig.from_dict(data)
        self.assertIn("Unsupported schema_version", str(ctx.exception))

    def test_reject_excessive_lag(self):
        """Lag >= training set size must raise ValueError."""
        data = PipelineConfig().to_dict()
        data["algorithm_settings"]["lag"] = 10
        with self.assertRaises(ValueError) as ctx:
            PipelineConfig.from_dict(data)
        self.assertIn("cannot be >= training set size", str(ctx.exception))


class TestInputShapeValidation(unittest.TestCase):

    def test_reject_missing_feature(self):
        """Input data missing configured feature must raise ValueError."""
        config = PipelineConfig(feature_ordering=["value"])
        model = AutoregressiveSequenceModel(config)
        bad_records = [{"record_id": "d1", "timestamp": "2026-01-01"}]  # missing 'value'
        with self.assertRaises(ValueError) as ctx:
            model.fit(bad_records)
        self.assertIn("Missing required feature", str(ctx.exception))

    def test_reject_non_numeric_feature(self):
        """Input with non-numeric feature must raise ValueError."""
        config = PipelineConfig(feature_ordering=["value"])
        model = AutoregressiveSequenceModel(config)
        bad_records = [
            {"record_id": "d1", "value": "not_a_number"},
            {"record_id": "d2", "value": 12.0},
        ]
        with self.assertRaises(ValueError) as ctx:
            model.fit(bad_records)
        self.assertIn("must be finite numeric", str(ctx.exception))

    def test_reject_missing_record_id(self):
        """Input record without record_id must raise ValueError."""
        config = PipelineConfig()
        model = AutoregressiveSequenceModel(config)
        bad_records = [{"value": 10.0}, {"value": 12.0}]
        with self.assertRaises(ValueError) as ctx:
            model.fit(bad_records)
        self.assertIn("missing required 'record_id'", str(ctx.exception))


class TestDeliberateTampering(unittest.TestCase):

    def test_tampered_predictions_fail(self):
        """Assert that tampered predictions do not match expected model predictions."""
        result = run_pipeline()
        tampered_test_preds = [0.0, 0.0]
        self.assertNotEqual(result["test_predictions"], tampered_test_preds)


if __name__ == "__main__":
    unittest.main()
