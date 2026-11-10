import unittest
from fixture import RECORDS, SCHEMA, SPLITS, get_split
from metrics import precision, recall, mae, leaky_relu, predict


class TestMetrics(unittest.TestCase):
    truth, pred = [1, 1, 0, 0], [1, 0, 1, 0]

    def test_precision_recall_binary(self):
        self.assertEqual(precision(self.truth, self.pred), 0.5)
        self.assertEqual(recall(self.truth, self.pred), 0.5)

    def test_macro_micro(self):
        self.assertEqual(precision(self.truth, self.pred, average="macro"), 0.5)
        self.assertEqual(recall(self.truth, self.pred, average="micro"), 0.5)

    def test_zero_denominator_precision(self):   # no predicted positives
        self.assertEqual(precision([1, 1], [0, 0]), 0.0)

    def test_zero_denominator_recall(self):      # no actual positives
        self.assertEqual(recall([0, 0], [0, 1]), 0.0)

    def test_zero_division_override(self):
        self.assertEqual(precision([1, 1], [0, 0], zero_division=1.0), 1.0)

    def test_mae(self):
        self.assertAlmostEqual(mae([1, 2, 3], [1, 3, 2]), 2 / 3)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            precision([1, 0], [1])
        with self.assertRaises(ValueError):
            recall([], [])
        with self.assertRaises(ValueError):
            precision([1], [1], average="weird")

    def test_leaky_relu(self):
        self.assertEqual(leaky_relu(5), 5)
        self.assertAlmostEqual(leaky_relu(-10), -2.0)


class TestFixture(unittest.TestCase):
    def test_schema_and_size(self):
        self.assertEqual(len(RECORDS), 6)
        for r in RECORDS:
            self.assertEqual(tuple(r.keys()), SCHEMA)
            self.assertEqual(len(r["pixels"]), 4)

    def test_unique_ids_and_labels(self):
        ids = [r["record_id"] for r in RECORDS]
        self.assertEqual(len(ids), len(set(ids)))
        labels = [r["label"] for r in RECORDS]
        self.assertEqual((labels.count(0), labels.count(1)), (3, 3))

    def test_splits_disjoint_by_group(self):
        seen = []
        for g in SPLITS.values():
            seen += g
        self.assertEqual(len(seen), len(set(seen)))
        ids = [r["record_id"] for s in SPLITS for r in get_split(s)]
        self.assertEqual(sorted(ids), sorted(r["record_id"] for r in RECORDS))

    def test_pipeline_end_to_end(self):          # deterministic, no RNG needed
        test = get_split("test")
        y = [r["label"] for r in test]
        p = [predict(r["pixels"]) for r in test]
        self.assertEqual(precision(y, p), 1.0)
        self.assertEqual(recall(y, p), 1.0)


if __name__ == "__main__":
    unittest.main()
