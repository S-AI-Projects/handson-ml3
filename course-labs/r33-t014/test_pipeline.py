import copy
import json
import unittest
from pathlib import Path

from pipeline import (
    classification_metrics,
    mean_absolute_error,
    validate_records,
    validate_splits,
    run_evaluation,
)

FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def load_fixture():
    with FIXTURE_PATH.open() as file:
        return json.load(file)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.data = load_fixture()

    def test_fixture_has_eight_records(self):
        self.assertEqual(len(self.data["records"]), 8)

    def test_fixture_values_match_contract(self):
        records = self.data["records"]
        self.assertEqual([r["x1"] for r in records], list(range(8)))
        self.assertEqual([r["x2"] for r in records], [1] * 8)
        self.assertEqual([r["label"] for r in records], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_record_and_group_ids_are_unique(self):
        records = self.data["records"]
        ids = [r["record_id"] for r in records]
        groups = [r["group_id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(groups), len(set(groups)))

    def test_fixture_schema_passes(self):
        self.assertEqual(validate_records(self.data["records"]), [])

    def test_fixed_stratified_split_passes(self):
        self.assertEqual(
            validate_splits(self.data["records"], self.data["splits"]), []
        )

    def test_each_split_has_expected_class_balance(self):
        by_id = {r["record_id"]: r["label"] for r in self.data["records"]}
        for name, ids in self.data["splits"].items():
            labels = [by_id[item] for item in ids]
            self.assertEqual(labels.count(0), labels.count(1))

    def test_classification_precision_and_recall(self):
        result = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_zero_precision_denominator_returns_zero(self):
        result = classification_metrics([1, 0], [0, 0])
        self.assertEqual(result["precision"], 0.0)

    def test_zero_recall_denominator_returns_zero(self):
        result = classification_metrics([0, 0], [1, 0])
        self.assertEqual(result["recall"], 0.0)

    def test_regression_mae(self):
        self.assertAlmostEqual(
            mean_absolute_error([1, 2, 3], [1, 3, 2]),
            2 / 3
        )

    def test_metric_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            mean_absolute_error([], [])

    def test_metric_rejects_different_lengths(self):
        with self.assertRaises(ValueError):
            classification_metrics([1, 0], [1])

    def test_metric_rejects_invalid_class_label(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 2], [0, 1])

    def test_duplicate_record_id_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[1]["record_id"] = records[0]["record_id"]
        self.assertTrue(any(
            e["field"] == "record_id" for e in validate_records(records)
        ))

    def test_invalid_label_is_rejected(self):
        records = copy.deepcopy(self.data["records"])
        records[0]["label"] = 3
        self.assertTrue(any(
            e["field"] == "label" for e in validate_records(records)
        ))

    def test_record_overlap_is_rejected(self):
        splits = copy.deepcopy(self.data["splits"])
        splits["test"].append(splits["train"][0])
        self.assertTrue(validate_splits(self.data["records"], splits))

    def test_concrete_evaluation_result(self):
        result = run_evaluation(self.data)
        self.assertEqual(result["classification"]["precision"], 0.5)
        self.assertEqual(result["classification"]["recall"], 0.5)
        self.assertAlmostEqual(result["regression_mae"], 2 / 3)
        self.assertEqual(result["fixture_record_count"], 8)


if __name__ == "__main__":
    unittest.main()
