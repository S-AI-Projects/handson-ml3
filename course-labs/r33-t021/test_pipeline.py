"""
Unit tests for Course Lab R33-T021 Deterministic Split Manifest.
Offline, CPU-only bounded verification.
"""

import json
import os
import unittest
# import via importlib below

# For unittest import compatibility
import importlib.util
pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
DeterministicSplitManifest = pipeline.DeterministicSplitManifest


class TestDeterministicSplitManifest(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        self.manifest_builder = DeterministicSplitManifest(seed=42)

    def test_fixture_integrity(self):
        """Asserts the 6 synthetic 4x4 image fixture properties."""
        self.assertEqual(len(self.fixture), 6)
        zeros = [r for r in self.fixture if r["label"] == 0]
        diags = [r for r in self.fixture if r["label"] == 1]
        self.assertEqual(len(zeros), 3)
        self.assertEqual(len(diags), 3)

        for r in zeros:
            self.assertEqual(len(r["pixels"]), 4)
            for row in r["pixels"]:
                self.assertEqual(row, [0, 0, 0, 0])

        for r in diags:
            self.assertEqual(len(r["pixels"]), 4)
            for i in range(4):
                self.assertEqual(r["pixels"][i][i], 255)

    def test_group_split_disjointness_and_determinism(self):
        """Asserts disjoint record sets and deterministic output across runs."""
        train_groups = ["grp_alpha", "grp_gamma"]
        val_groups = ["grp_beta"]
        test_groups = ["grp_delta"]

        manifest1 = self.manifest_builder.split_by_group(self.fixture, train_groups, val_groups, test_groups)
        manifest2 = self.manifest_builder.split_by_group(self.fixture, train_groups, val_groups, test_groups)

        self.assertEqual(manifest1, manifest2)
        self.assertEqual(manifest1["train"], ["img_001", "img_002", "img_004"])
        self.assertEqual(manifest1["val"], ["img_003"])
        self.assertEqual(manifest1["test"], ["img_005", "img_006"])

        s_train = set(manifest1["train"])
        s_val = set(manifest1["val"])
        s_test = set(manifest1["test"])
        self.assertTrue(s_train.isdisjoint(s_val))
        self.assertTrue(s_train.isdisjoint(s_test))
        self.assertTrue(s_val.isdisjoint(s_test))

    def test_overlapping_groups_rejected(self):
        """Asserts intentionally overlapping groups are rejected."""
        with self.assertRaises(ValueError):
            self.manifest_builder.split_by_group(
                self.fixture, 
                train_groups=["grp_alpha"], 
                val_groups=["grp_alpha"], 
                test_groups=["grp_delta"]
            )

    def test_chronological_time_series_split(self):
        """Asserts chronological splits follow temporal order."""
        manifest = self.manifest_builder.split_chronological(self.fixture, train_ratio=0.5, val_ratio=0.25)
        self.assertEqual(manifest["train"], ["img_001", "img_002", "img_003"])
        self.assertEqual(manifest["val"], ["img_004"])
        self.assertEqual(manifest["test"], ["img_005", "img_006"])

    def test_deliberate_mutation_failure(self):
        """Asserts that deliberately altering expected results fails."""
        manifest = self.manifest_builder.split_by_group(
            self.fixture, 
            train_groups=["grp_alpha", "grp_gamma"], 
            val_groups=["grp_beta"], 
            test_groups=["grp_delta"]
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(manifest["train"], ["img_001", "img_MUTATED"])


if __name__ == "__main__":
    unittest.main()
