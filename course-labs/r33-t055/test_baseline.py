import unittest
from fixture import RECORDS, SCHEMA, SPLITS, get_split
from baseline import fit_majority, predict_majority, accuracy

EXPECTED_TEST_ACCURACY = 0.5


def labels(split):
    return [r["label"] for r in get_split(split)]


class TestBaseline(unittest.TestCase):
    def test_majority_clear_winner(self):
        self.assertEqual(fit_majority([0, 0, 1]), 0)
        self.assertEqual(fit_majority([1, 1, 0]), 1)

    def test_majority_tie_goes_to_smallest_label(self):
        self.assertEqual(fit_majority([1, 0, 1, 0]), 0)

    def test_fit_on_train_split(self):
        self.assertEqual(fit_majority(labels("train")), 0)

    def test_heldout_accuracy(self):
        m = fit_majority(labels("train"))
        y = labels("test")
        self.assertEqual(accuracy(y, predict_majority(m, len(y))), EXPECTED_TEST_ACCURACY)

    def test_val_accuracy_and_prediction_count(self):
        m = fit_majority(labels("train"))
        y = labels("val")
        p = predict_majority(m, len(y))
        self.assertEqual(len(p), 2)
        self.assertEqual(accuracy(y, p), 0.5)

    def test_baseline_uses_only_training_data(self):
        # Flipping every val/test label must not change the fitted baseline.
        base = fit_majority(labels("train"))
        flipped = [1 - r["label"] for s in ("val", "test") for r in get_split(s)]
        self.assertEqual(len(flipped), 4)
        self.assertEqual(fit_majority(labels("train")), base)

    def test_accuracy_perfect_and_zero(self):
        self.assertEqual(accuracy([0, 1], [0, 1]), 1.0)
        self.assertEqual(accuracy([0, 1], [1, 0]), 0.0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fit_majority([])
        with self.assertRaises(ValueError):
            accuracy([1, 0], [1])
        with self.assertRaises(ValueError):
            accuracy([], [])
        with self.assertRaises(ValueError):
            predict_majority(0, -1)


class TestFixture(unittest.TestCase):
    def test_schema_and_values(self):
        self.assertEqual(len(RECORDS), 8)
        for r in RECORDS:
            self.assertEqual(tuple(r.keys()), SCHEMA)
            self.assertEqual(r["x2"], 1)
        self.assertEqual([r["x1"] for r in RECORDS], list(range(8)))
        self.assertEqual([r["label"] for r in RECORDS], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_unique_ids(self):
        ids = [r["record_id"] for r in RECORDS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_splits_disjoint_and_complete(self):
        seen = [g for gs in SPLITS.values() for g in gs]
        self.assertEqual(len(seen), len(set(seen)))
        ids = [r["record_id"] for s in SPLITS for r in get_split(s)]
        self.assertEqual(sorted(ids), sorted(r["record_id"] for r in RECORDS))

    def test_split_is_stratified(self):
        for s in SPLITS:
            ls = labels(s)
            self.assertEqual(ls.count(0), ls.count(1))


if __name__ == "__main__":
    unittest.main()
