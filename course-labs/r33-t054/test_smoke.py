import math
import unittest
from fixture import RECORDS, SCHEMA, SPLITS, get_split
from smoke import train, predict_proba, scalar_loss, MAX_SAMPLES, MAX_STEPS

EXPECTED_STEPS = 20


class TestSmokeRun(unittest.TestCase):
    def setUp(self):
        self.train_set = get_split("train")
        self.result = train(self.train_set)

    def test_recorded_step_count(self):
        self.assertEqual(self.result["steps"], EXPECTED_STEPS)
        self.assertEqual(len(self.result["losses"]), EXPECTED_STEPS)

    def test_loss_is_finite(self):
        for l in self.result["losses"]:
            self.assertTrue(math.isfinite(l))

    def test_prediction_shape(self):
        for split in SPLITS:
            recs = get_split(split)
            p = predict_proba(self.result["w"], self.result["b"], recs)
            self.assertEqual(len(p), len(recs))
            for v in p:
                self.assertTrue(0.0 <= v <= 1.0)

    def test_deterministic_with_fixed_seed(self):
        again = train(self.train_set)
        self.assertEqual(self.result["losses"], again["losses"])

    def test_scalar_loss_is_scalar(self):
        v = scalar_loss(1, 0.5)
        self.assertIsInstance(v, float)
        self.assertAlmostEqual(v, math.log(2))

    def test_scalar_loss_clamps_extremes(self):
        self.assertTrue(math.isfinite(scalar_loss(1, 0.0)))

    def test_too_many_steps_rejected(self):
        with self.assertRaises(ValueError):
            train(self.train_set, steps=MAX_STEPS + 1)

    def test_too_many_samples_rejected(self):
        with self.assertRaises(ValueError):
            train(RECORDS * 6)           # 36 > 32

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            train([])

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"),
                         "tensorflow not installed")
    def test_tf_scalar_shape(self):
        import tensorflow as tf
        self.assertEqual(tf.constant(42).shape, ())


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

    def test_splits_disjoint_by_group(self):
        seen = [g for gs in SPLITS.values() for g in gs]
        self.assertEqual(len(seen), len(set(seen)))
        ids = [r["record_id"] for s in SPLITS for r in get_split(s)]
        self.assertEqual(sorted(ids), sorted(r["record_id"] for r in RECORDS))


if __name__ == "__main__":
    unittest.main()
