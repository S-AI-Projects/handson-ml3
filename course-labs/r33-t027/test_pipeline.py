import unittest
import json
import os
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

ReproduciblePipelineConfig = pipeline.ReproduciblePipelineConfig
ConfiguredClassifierPipeline = pipeline.ConfiguredClassifierPipeline

class TestReproducibleConfiguration(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        self.config = ReproduciblePipelineConfig()

    def test_fixture_integrity(self):
        """Asserts 8 synthetic rows x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.fixture), 8)
        self.assertEqual([r["x1"] for r in self.fixture], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual([r["label"] for r in self.fixture], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_config_serialization_roundtrip(self):
        """Reloaded config reconstructs the exact same split IDs and predictions."""
        config_json = self.config.to_json()
        reloaded = ReproduciblePipelineConfig.from_json(config_json)

        self.assertEqual(self.config.split_ids, reloaded.split_ids)
        self.assertEqual(self.config.seed, reloaded.seed)

        pipe_orig = ConfiguredClassifierPipeline(self.config)
        pipe_reloaded = ConfiguredClassifierPipeline(reloaded)

        preds_orig = pipe_orig.predict(self.fixture)
        preds_reloaded = pipe_reloaded.predict(self.fixture)
        self.assertEqual(preds_orig, preds_reloaded)
        self.assertEqual(preds_orig, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_unknown_field_rejected(self):
        """Unknown configuration fields must be rejected."""
        bad_data = {
            "seed": 42,
            "schema_version": "1.0",
            "feature_ordering": ["x1", "x2"],
            "split_ids": {"train": ["row_0"], "val": [], "test": []},
            "algorithm_settings": {"threshold": 3.5},
            "unsupported_extra_field": "invalid"
        }
        with self.assertRaises(ValueError):
            ReproduciblePipelineConfig.from_dict(bad_data)

    def test_contradictory_settings_rejected(self):
        """Overlapping split IDs are rejected."""
        with self.assertRaises(ValueError):
            ReproduciblePipelineConfig(
                split_ids={"train": ["row_0"], "val": ["row_0"], "test": []}
            )

    def test_incompatible_input_shape_rejected(self):
        """Missing expected features raises ValueError."""
        pipe = ConfiguredClassifierPipeline(self.config)
        malformed = [{"record_id": "bad1", "x1": 2.0}]  # missing x2
        with self.assertRaises(ValueError):
            pipe.predict(malformed)

    def test_deliberate_mutation_failure(self):
        """Mutating predicted threshold output fails check."""
        pipe = ConfiguredClassifierPipeline(self.config)
        preds = pipe.predict(self.fixture)
        mutated_preds = [1]*8
        self.assertNotEqual(preds, mutated_preds)
