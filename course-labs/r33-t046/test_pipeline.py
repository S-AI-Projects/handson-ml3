"""
Unit tests for Course Lab R33-T046 Decision Tree Configuration Schema.
"""
import unittest
import json
import os
from pipeline import DecisionTreeConfig

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestDecisionTreeConfig(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)
        self.valid_dict = {
            "schema_version": "1.0.0",
            "random_seed": 42,
            "feature_ordering": ["x1", "x2"],
            "split_ids": {
                "train": ["r1", "r2", "r5", "r6"],
                "validation": ["r3", "r7"],
                "test": ["r4", "r8"]
            },
            "algorithm_settings": {
                "max_depth": 3,
                "criterion": "gini",
                "min_samples_split": 2
            }
        }

    def test_round_trip_serialization(self):
        cfg = DecisionTreeConfig.from_dict(self.valid_dict)
        json_str = cfg.to_json()
        cfg2 = DecisionTreeConfig.from_json(json_str)
        self.assertEqual(cfg.to_dict(), cfg2.to_dict())

    def test_stratified_split_balance(self):
        cfg = DecisionTreeConfig.from_dict(self.valid_dict)
        rec_map = {r["record_id"]: r["label"] for r in self.fixture}
        train_labels = [rec_map[rid] for rid in cfg.split_ids["train"]]
        self.assertEqual(train_labels.count(0), 2)
        self.assertEqual(train_labels.count(1), 2)

    def test_reject_unknown_field(self):
        bad = dict(self.valid_dict)
        bad["unauthorized_flag"] = True
        with self.assertRaises(ValueError) as ctx:
            DecisionTreeConfig.from_dict(bad)
        self.assertIn("Unknown configuration keys", str(ctx.exception))

    def test_reject_contradictory_max_depth(self):
        bad = json.loads(json.dumps(self.valid_dict))
        bad["algorithm_settings"]["max_depth"] = -5
        with self.assertRaises(ValueError):
            DecisionTreeConfig.from_dict(bad)

    def test_reject_unknown_criterion(self):
        bad = json.loads(json.dumps(self.valid_dict))
        bad["algorithm_settings"]["criterion"] = "magic_loss"
        with self.assertRaises(ValueError):
            DecisionTreeConfig.from_dict(bad)

    def test_reject_split_overlap(self):
        bad = json.loads(json.dumps(self.valid_dict))
        bad["split_ids"]["train"].append("r4")  # r4 also in test
        with self.assertRaises(ValueError):
            DecisionTreeConfig.from_dict(bad)

if __name__ == "__main__":
    unittest.main()
