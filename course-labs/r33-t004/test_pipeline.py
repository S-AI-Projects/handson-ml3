"""
Unit test suite for Course Lab R33-T004 metric correctness.
Verifies exact canonical classification & regression metrics, zero-denominator handling,
and 8-row decision tree fixture evaluation.
"""

from pathlib import Path
import unittest

from pipeline import (
    ClassificationMetrics,
    DecisionStumpClassifier,
    RegressionMetrics,
    calculate_classification_metrics,
    calculate_regression_metrics,
    compute_binary_confusion_matrix,
    fixed_stratified_split,
    load_dataset,
)


class TestR33T004MetricCorrectness(unittest.TestCase):
    """Offline, CPU-only unit tests for metric correctness."""

    def setUp(self) -> None:
        self.lab_dir = Path(__file__).parent
        self.json_path = self.lab_dir / "fixture.json"
        self.csv_path = self.lab_dir / "fixture.csv"
        self.fixture = load_dataset(self.json_path)

    # 1. Canonical Classification Benchmark
    def test_canonical_classification_precision_recall_half(self) -> None:
        """
        Required canonical case:
        truth=[1, 1, 0, 0] / prediction=[1, 0, 1, 0] -> precision=recall=0.5, accuracy=0.5, F1=0.5.
        TP=1, FP=1, TN=1, FN=1.
        """
        truth = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]

        tp, fp, tn, fn = compute_binary_confusion_matrix(truth, prediction, pos_label=1)
        self.assertEqual((tp, fp, tn, fn), (1, 1, 1, 1))

        metrics = calculate_classification_metrics(truth, prediction)
        self.assertAlmostEqual(metrics.precision, 0.5, places=7)
        self.assertAlmostEqual(metrics.recall, 0.5, places=7)
        self.assertAlmostEqual(metrics.accuracy, 0.5, places=7)
        self.assertAlmostEqual(metrics.f1_score, 0.5, places=7)
        self.assertFalse(metrics.zero_division_applied)

    # 2. Canonical Regression Benchmark
    def test_canonical_regression_mae_two_thirds(self) -> None:
        """
        Required canonical case:
        truth=[1, 2, 3] / prediction=[1, 3, 2] -> MAE = 2/3 (0.666666...).
        """
        truth = [1.0, 2.0, 3.0]
        prediction = [1.0, 3.0, 2.0]

        reg_metrics = calculate_regression_metrics(truth, prediction)
        expected_mae = 2.0 / 3.0

        self.assertAlmostEqual(reg_metrics.mae, expected_mae, places=7)
        self.assertAlmostEqual(reg_metrics.mse, expected_mae, places=7)

    # 3. Explicit Zero-Denominator Handling
    def test_zero_denominator_no_positive_predictions(self) -> None:
        """
        When model predicts all zeroes, TP + FP = 0.
        Precision denominator is zero; must explicitly return zero_division (default 0.0).
        """
        truth = [1, 1, 0, 0]
        all_zero_preds = [0, 0, 0, 0]

        metrics = calculate_classification_metrics(truth, all_zero_preds, zero_division=0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertTrue(metrics.zero_division_applied)

    def test_zero_denominator_no_positive_ground_truth(self) -> None:
        """When ground truth has no positive labels, TP + FN = 0; recall denominator is zero."""
        truth = [0, 0, 0, 0]
        preds = [0, 0, 0, 0]

        metrics = calculate_classification_metrics(truth, preds, zero_division=0.0)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertTrue(metrics.zero_division_applied)

    # 4. Exact 8-row Fixture & Stratified Split
    def test_synthetic_fixture_dimensions_and_contract(self) -> None:
        """Fixture must have 8 rows with x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.fixture), 8)

        expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_x2 = [1] * 8
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]

        self.assertEqual([r["x1"] for r in self.fixture], expected_x1)
        self.assertEqual([r["x2"] for r in self.fixture], expected_x2)
        self.assertEqual([r["label"] for r in self.fixture], expected_labels)

        # Check unique record IDs and stable group IDs
        record_ids = [r["record_id"] for r in self.fixture]
        self.assertEqual(len(record_ids), len(set(record_ids)))

    def test_csv_fixture_loading_parity(self) -> None:
        """CSV fixture must load identically to JSON fixture."""
        csv_records = load_dataset(self.csv_path)
        self.assertEqual(csv_records, self.fixture)

    def test_stratified_split_preserves_class_balance(self) -> None:
        """Stratified split (50/50) must yield 2 zeroes and 2 ones in each split."""
        train, test = fixed_stratified_split(self.fixture, train_ratio=0.5)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 4)

        train_labels = [r["label"] for r in train]
        test_labels = [r["label"] for r in test]

        self.assertEqual(train_labels.count(0), 2)
        self.assertEqual(train_labels.count(1), 2)
        self.assertEqual(test_labels.count(0), 2)
        self.assertEqual(test_labels.count(1), 2)

    # 5. Decision Tree Stump Regularization Evaluation
    def test_decision_stump_perfect_separator(self) -> None:
        """A decision stump with threshold 3.5 separates the 8-row fixture with 100% precision/recall."""
        stump = DecisionStumpClassifier(threshold=3.5)
        eval_metrics = stump.evaluate(self.fixture)

        self.assertEqual(eval_metrics.accuracy, 1.0)
        self.assertEqual(eval_metrics.precision, 1.0)
        self.assertEqual(eval_metrics.recall, 1.0)
        self.assertEqual(eval_metrics.f1_score, 1.0)
        self.assertEqual(eval_metrics.tp, 4)
        self.assertEqual(eval_metrics.tn, 4)

    def test_decision_stump_regularized_suboptimal(self) -> None:
        """A regularized stump with threshold 5.5 misses samples x1=4 and x1=5 (FN=2)."""
        stump = DecisionStumpClassifier(threshold=5.5)
        eval_metrics = stump.evaluate(self.fixture)

        # Predicts 1 only for x1 >= 5.5 -> {6, 7} (2 samples)
        # Truth has 4 positives -> TP=2, FN=2, FP=0, TN=4
        self.assertEqual(eval_metrics.tp, 2)
        self.assertEqual(eval_metrics.fn, 2)
        self.assertEqual(eval_metrics.fp, 0)
        self.assertEqual(eval_metrics.precision, 1.0)  # 2 / (2 + 0)
        self.assertEqual(eval_metrics.recall, 0.5)     # 2 / (2 + 2)


if __name__ == "__main__":
    unittest.main()
