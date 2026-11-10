import copy
import json
import os
import tempfile
import unittest

from config import (FEATURE_ORDER, ConfigError, environment_info, extract_features, fit,
                    load_config, make_config, predict, save_config, splits_from_config,
                    validate_config)
from fixture import RECORDS, SCHEMA

SEED = 0
EXPECTED_SEED = 0
TOL = 1e-9


def split_ids(cfg):
    return {k: [r["record_id"] for r in v] for k, v in splits_from_config(cfg, RECORDS).items()}


class TestConfig(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "config.json")
        self.cfg = make_config(RECORDS, seed=SEED)
        save_config(self.cfg, self.path)

    def _bad(self, mutate):
        cfg = copy.deepcopy(self.cfg)
        mutate(cfg)
        return cfg

    def test_reloaded_config_reconstructs_same_split(self):
        loaded = load_config(self.path, RECORDS)
        self.assertEqual(loaded, self.cfg)
        self.assertEqual(split_ids(loaded), split_ids(self.cfg))

    def test_make_config_is_deterministic(self):
        self.assertEqual(make_config(RECORDS, seed=SEED), self.cfg)
        self.assertEqual(make_config(RECORDS, seed=7), make_config(RECORDS, seed=7))

    def test_reloaded_predictions_identical(self):
        loaded = load_config(self.path, RECORDS)
        c1, c2 = fit(self.cfg, RECORDS), fit(loaded, RECORDS)
        for r in RECORDS:
            p1, p2 = predict(self.cfg, c1, r["pixels"]), predict(loaded, c2, r["pixels"])
            self.assertEqual(p1["label"], p2["label"])
            for k in p1["scores"]:
                self.assertLessEqual(abs(p1["scores"][k] - p2["scores"][k]), TOL)

    def test_predictions_match_labels_on_test_split(self):
        cents = fit(self.cfg, RECORDS)
        for r in splits_from_config(self.cfg, RECORDS)["test"]:
            self.assertEqual(predict(self.cfg, cents, r["pixels"])["label"], r["label"])

    def test_splits_group_disjoint_and_complete(self):
        s = splits_from_config(self.cfg, RECORDS)
        groups = {k: {r["group_id"] for r in v} for k, v in s.items()}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        ids = sorted(r["record_id"] for v in s.values() for r in v)
        self.assertEqual(ids, sorted(r["record_id"] for r in RECORDS))

    def test_unknown_field_rejected(self):
        with self.assertRaisesRegex(ConfigError, "unknown"):
            validate_config(self._bad(lambda c: c.update(extra=1)))
        with self.assertRaisesRegex(ConfigError, "unknown"):
            validate_config(self._bad(lambda c: c["algorithm"].update(kernel="rbf")))

    def test_missing_field_rejected(self):
        with self.assertRaisesRegex(ConfigError, "missing"):
            validate_config(self._bad(lambda c: c.pop("seed")))

    def test_wrong_schema_version_rejected(self):
        with self.assertRaisesRegex(ConfigError, "schema_version"):
            validate_config(self._bad(lambda c: c.update(schema_version=2)))

    def test_overlapping_split_ids_rejected(self):
        bad = self._bad(lambda c: c["split_ids"]["val"].append(c["split_ids"]["train"][0]))
        with self.assertRaisesRegex(ConfigError, "more than one split"):
            validate_config(bad)

    def test_group_leakage_rejected(self):
        def move(c):
            c["split_ids"]["val"].append(c["split_ids"]["train"].pop())
        with self.assertRaisesRegex(ConfigError, "group"):
            validate_config(self._bad(move), RECORDS)

    def test_unknown_record_id_rejected(self):
        bad = self._bad(lambda c: c["split_ids"]["train"].append("r99"))
        with self.assertRaisesRegex(ConfigError, "unknown record"):
            validate_config(bad, RECORDS)

    def test_bad_feature_order_rejected(self):
        dup = self._bad(lambda c: c["feature_order"].__setitem__(1, c["feature_order"][0]))
        with self.assertRaisesRegex(ConfigError, "feature_order"):
            validate_config(dup)
        short = self._bad(lambda c: c["feature_order"].pop())
        with self.assertRaisesRegex(ConfigError, "feature_order"):
            validate_config(short)

    def test_bad_algorithm_rejected(self):
        with self.assertRaisesRegex(ConfigError, "algorithm"):
            validate_config(self._bad(lambda c: c["algorithm"].update(name="svm")))
        with self.assertRaisesRegex(ConfigError, "pixel_max"):
            validate_config(self._bad(lambda c: c["algorithm"].update(pixel_max=1)))

    def test_bad_seed_rejected(self):
        for bad_seed in (-1, True, "0", 1.5):
            with self.assertRaisesRegex(ConfigError, "seed"):
                validate_config(self._bad(lambda c: c.update(seed=bad_seed)))

    def test_incompatible_input_shape_rejected(self):
        with self.assertRaises(ConfigError):
            extract_features(self.cfg, [[0] * 4] * 3)
        with self.assertRaises(ConfigError):
            extract_features(self.cfg, [[0] * 5] * 4)
        with self.assertRaisesRegex(ConfigError, "input_shape"):
            validate_config(self._bad(lambda c: c.update(input_shape=[3, 4])))

    def test_feature_order_applied(self):
        img = [[0] * 4 for _ in range(4)]
        img[0][0] = 255
        forward = extract_features(self.cfg, img)
        rev_cfg = self._bad(lambda c: c.update(feature_order=list(reversed(FEATURE_ORDER))))
        validate_config(rev_cfg)
        reverse = extract_features(rev_cfg, img)
        self.assertEqual((forward[0], forward[-1]), (1.0, 0.0))
        self.assertEqual((reverse[0], reverse[-1]), (0.0, 1.0))

    def test_environment_recorded(self):
        info = environment_info()
        for key in ("python", "implementation", "platform"):
            self.assertTrue(info[key])

    def test_saved_file_seed(self):
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["seed"], EXPECTED_SEED)

    def test_invalid_json_rejected(self):
        p = os.path.join(self.dir, "garbage.json")
        with open(p, "w") as f:
            f.write("not json at all")
        with self.assertRaises(ConfigError):
            load_config(p)


class TestFixture(unittest.TestCase):
    def test_schema_and_size(self):
        self.assertEqual(len(RECORDS), 6)
        for r in RECORDS:
            self.assertEqual(tuple(r.keys()), SCHEMA)

    def test_unique_ids_and_labels(self):
        ids = [r["record_id"] for r in RECORDS]
        self.assertEqual(len(ids), len(set(ids)))
        labels = [r["label"] for r in RECORDS]
        self.assertEqual((labels.count(0), labels.count(1)), (3, 3))

    def test_groups_declared(self):
        self.assertEqual(sorted({r["group_id"] for r in RECORDS}), ["g1", "g2", "g3"])


if __name__ == "__main__":
    unittest.main()
