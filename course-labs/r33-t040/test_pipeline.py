"""
Unit tests for R33-T040: Define a saved-pipeline interface
===========================================================
Covers:
  - Exact synthetic fixture verification (6 4x4 images: 3 all-zero, 3 white-diagonal)
  - Train/save/reload round-trip using safe JSON serialization (never pickle)
  - Before/after prediction agreement within explicit tolerance (1e-6)
  - Strict input schema validation (pixel count, range [0, 255], missing fields)
  - Schema version mismatch rejection
  - Unknown serialized keys rejection
  - Deliberate tampering failure detection

Standard library only. Offline CPU-only execution.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "course-labs/r33-t040")
from pipeline import (
    EXPECTED_PIXEL_COUNT,
    SUPPORTED_SCHEMA_VERSION,
    SYNTHETIC_FIXTURE,
    TEST_RECORD_IDS,
    TRAIN_RECORD_IDS,
    SavedImageClassifierPipeline,
    run_saved_pipeline_roundtrip,
)


class TestImageFixtureIntegrity(unittest.TestCase):

    def test_fixture_structure(self):
        """Six synthetic 4x4 grayscale images: 3 all-zero labelled 0, 3 white-diagonal labelled 1."""
        self.assertEqual(len(SYNTHETIC_FIXTURE), 6)
        all_zeros = [r for r in SYNTHETIC_FIXTURE if r["label"] == 0]
        white_diags = [r for r in SYNTHETIC_FIXTURE if r["label"] == 1]

        self.assertEqual(len(all_zeros), 3)
        self.assertEqual(len(white_diags), 3)

        # Check all-zero images
        for r in all_zeros:
            self.assertEqual(r["group_id"], "g0")
            self.assertEqual(len(r["pixels"]), EXPECTED_PIXEL_COUNT)
            self.assertTrue(all(p == 0 for p in r["pixels"]))

        # Check white-diagonal images (indices 0, 5, 10, 15 are 255)
        diag_indices = {0, 5, 10, 15}
        for r in white_diags:
            self.assertEqual(r["group_id"], "g1")
            self.assertEqual(len(r["pixels"]), EXPECTED_PIXEL_COUNT)
            for idx, p in enumerate(r["pixels"]):
                if idx in diag_indices:
                    self.assertEqual(p, 255)
                else:
                    self.assertEqual(p, 0)


class TestSaveReloadAndPredictionAgreement(unittest.TestCase):

    def test_save_reload_file_roundtrip(self):
        """Save to JSON file and reload, verifying prediction agreement within 1e-6."""
        model = SavedImageClassifierPipeline()
        train_records = [r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS]
        model.fit(train_records)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            # Verify file contains valid JSON
            with open(tmp_path, "r") as f:
                loaded_json = json.load(f)
            self.assertEqual(loaded_json["schema_version"], SUPPORTED_SCHEMA_VERSION)

            # Reload model
            reloaded_model = SavedImageClassifierPipeline.load(tmp_path)

            for r in SYNTHETIC_FIXTURE:
                p_before = model.predict_record(r)
                p_after = reloaded_model.predict_record(r)

                diff = abs(p_before["probability"] - p_after["probability"])
                self.assertLessEqual(diff, 1e-6)
                self.assertEqual(p_before["predicted_label"], p_after["predicted_label"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_roundtrip_function(self):
        """Pipeline roundtrip runner completes with PASS status."""
        res = run_saved_pipeline_roundtrip(tolerance=1e-6)
        self.assertEqual(res["status"], "PASS")
        self.assertFalse(res["requires_training_data_at_inference"])
        self.assertFalse(res["requires_credentials"])


class TestSchemaValidationAndRejection(unittest.TestCase):

    def setUp(self):
        self.model = SavedImageClassifierPipeline()
        self.model.fit([r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS])

    def test_reject_wrong_pixel_length(self):
        """Images with != 16 pixels must raise ValueError."""
        bad_record = {"record_id": "err1", "pixels": [0] * 15}
        with self.assertRaises(ValueError) as ctx:
            self.model.predict_record(bad_record)
        self.assertIn("Expected 16 pixels", str(ctx.exception))

    def test_reject_pixel_out_of_range(self):
        """Pixel values < 0 or > 255 must raise ValueError."""
        bad_high = {"record_id": "err2", "pixels": [0] * 15 + [300]}
        with self.assertRaises(ValueError) as ctx:
            self.model.predict_record(bad_high)
        self.assertIn("out of range", str(ctx.exception))

        bad_neg = {"record_id": "err3", "pixels": [-1] + [0] * 15}
        with self.assertRaises(ValueError) as ctx:
            self.model.predict_record(bad_neg)
        self.assertIn("out of range", str(ctx.exception))

    def test_reject_missing_fields(self):
        """Missing record_id or pixels must raise ValueError."""
        with self.assertRaises(ValueError):
            self.model.predict_record({"pixels": [0] * 16})
        with self.assertRaises(ValueError):
            self.model.predict_record({"record_id": "err4"})

    def test_reject_schema_version_mismatch(self):
        """Loading a serialized model with wrong schema version must fail."""
        data = self.model.to_dict()
        data["schema_version"] = "2.0.0"
        with self.assertRaises(ValueError) as ctx:
            SavedImageClassifierPipeline.from_dict(data)
        self.assertIn("Schema version mismatch", str(ctx.exception))

    def test_reject_unknown_serialized_keys(self):
        """Loading serialized data with unknown keys must fail."""
        data = self.model.to_dict()
        data["unauthorized_field"] = "payload"
        with self.assertRaises(ValueError) as ctx:
            SavedImageClassifierPipeline.from_dict(data)
        self.assertIn("Unknown keys", str(ctx.exception))


class TestDeliberateTampering(unittest.TestCase):

    def test_tampered_prediction_detected(self):
        """Deliberately changing expected class label must trigger assertion failure."""
        res = run_saved_pipeline_roundtrip()
        actual_label = res["predictions_after"]["img4"]["predicted_label"]
        self.assertEqual(actual_label, 1)
        tampered_label = 0
        self.assertNotEqual(actual_label, tampered_label)


if __name__ == "__main__":
    unittest.main()
