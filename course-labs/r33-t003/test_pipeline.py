"""
Unit test suite for Course Lab R33-T003 majority-class baseline.
Tests 4x4 image fixture, baseline training/prediction, metric evaluation on held-out data,
and schema error handling.
"""

from pathlib import Path
import unittest

from pipeline import (
    MajorityClassBaseline,
    NotFittedError,
    SchemaError,
    load_dataset,
    split_train_test,
    validate_dataset,
)


class TestR33T003MajorityBaseline(unittest.TestCase):
    """Offline, CPU-only unit tests for minimal majority-class image baseline."""

    def setUp(self) -> None:
        self.lab_dir = Path(__file__).parent
        self.json_path = self.lab_dir / "fixture.json"
        self.csv_path = self.lab_dir / "fixture.csv"
        self.fixture = load_dataset(self.json_path)

    # 1. Exact Synthetic Fixture Validation
    def test_fixture_exact_dimensions_and_patterns(self) -> None:
        """Fixture must contain 6 synthetic 4x4 images: 3 all-zero (label 0) and 3 diagonal (label 1)."""
        self.assertEqual(len(self.fixture), 6)

        zero_records = [r for r in self.fixture if r["label"] == 0]
        diag_records = [r for r in self.fixture if r["label"] == 1]

        self.assertEqual(len(zero_records), 3, "Expected 3 zero-image samples.")
        self.assertEqual(len(diag_records), 3, "Expected 3 diagonal-image samples.")

        # Check all-zero pixel vectors
        for r in zero_records:
            self.assertEqual(r["pixels"], [0] * 16)
            self.assertIn("record_id", r)
            self.assertIn("group_id", r)

        # Check diagonal pixel vectors (indices 0, 5, 10, 15 are 255; others 0)
        expected_diagonal = [
            255 if idx in (0, 5, 10, 15) else 0
            for idx in range(16)
        ]
        for r in diag_records:
            self.assertEqual(r["pixels"], expected_diagonal)
            self.assertIn("record_id", r)
            self.assertIn("group_id", r)

    def test_csv_fixture_identical_loading(self) -> None:
        """CSV fixture must load identically to JSON fixture."""
        csv_records = load_dataset(self.csv_path)
        self.assertEqual(len(csv_records), 6)
        self.assertEqual(csv_records, self.fixture)

    # 2. Baseline Training (Uses ONLY Training Data)
    def test_majority_baseline_learns_only_from_training_data(self) -> None:
        """
        When training on [img_01, img_02, img_03] (all label 0) + [img_04] (label 1),
        majority class must be 0 based solely on training set counts.
        """
        train_set, test_set = split_train_test(
            self.fixture,
            train_ids=["img_01", "img_02", "img_03", "img_04"],
            test_ids=["img_05", "img_06"],
        )

        model = MajorityClassBaseline()
        self.assertFalse(model.is_fitted)

        model.fit(train_set)
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.majority_class, 0)
        self.assertEqual(model.train_class_counts, {0: 3, 1: 1})
        self.assertEqual(model.train_sample_count, 4)

    # 3. Held-out Evaluation on Test Partition
    def test_held_out_evaluation_metric_accuracy(self) -> None:
        """
        Evaluating majority class 0 on test set [img_05, img_06] (both label 1)
        yields exactly 0% accuracy (0/2 correct), accurately reflecting baseline performance.
        """
        train_set, test_set = split_train_test(
            self.fixture,
            train_ids=["img_01", "img_02", "img_03", "img_04"],
            test_ids=["img_05", "img_06"],
        )

        model = MajorityClassBaseline()
        model.fit(train_set)

        predictions = model.predict(test_set)
        self.assertEqual(predictions, [0, 0])

        eval_result = model.evaluate(test_set)
        self.assertEqual(eval_result.majority_class, 0)
        self.assertEqual(eval_result.test_sample_count, 2)
        self.assertEqual(eval_result.correct_predictions, 0)
        self.assertEqual(eval_result.accuracy, 0.0)

    def test_balanced_split_evaluation(self) -> None:
        """Balanced split test (1 zero, 1 diagonal in test)."""
        train_set, test_set = split_train_test(
            self.fixture,
            train_ids=["img_01", "img_02", "img_04"],  # 2 zeroes, 1 diagonal -> majority 0
            test_ids=["img_03", "img_05"],  # 1 zero, 1 diagonal
        )

        model = MajorityClassBaseline()
        model.fit(train_set)
        eval_result = model.evaluate(test_set)

        self.assertEqual(model.majority_class, 0)
        self.assertEqual(eval_result.correct_predictions, 1)
        self.assertEqual(eval_result.accuracy, 0.5)

    # 4. Error Handling & Edge Cases
    def test_predict_before_fit_raises_error(self) -> None:
        """Calling predict or evaluate before fit must raise NotFittedError."""
        model = MajorityClassBaseline()
        with self.assertRaises(NotFittedError):
            model.predict(self.fixture)

        with self.assertRaises(NotFittedError):
            model.evaluate(self.fixture)

    def test_malformed_pixel_dimensions_rejected(self) -> None:
        """Non-16 pixel count must raise SchemaError."""
        bad_record = [{"record_id": "bad", "group_id": "g1", "label": 0, "pixels": [0, 0, 0]}]
        with self.assertRaises(SchemaError):
            validate_dataset(bad_record)

    def test_out_of_range_pixel_values_rejected(self) -> None:
        """Pixel values outside [0, 255] or negative must raise SchemaError."""
        bad_record = [{"record_id": "bad", "group_id": "g1", "label": 0, "pixels": [-1] * 16}]
        with self.assertRaises(SchemaError):
            validate_dataset(bad_record)

    def test_invalid_label_rejected(self) -> None:
        """Labels other than binary 0 or 1 must raise SchemaError."""
        bad_record = [{"record_id": "bad", "group_id": "g1", "label": 2, "pixels": [0] * 16}]
        with self.assertRaises(SchemaError):
            validate_dataset(bad_record)

    def test_overlapping_train_test_split_rejected(self) -> None:
        """Intentional overlap in split IDs must raise ValueError."""
        with self.assertRaises(ValueError):
            split_train_test(self.fixture, train_ids=["img_01", "img_02"], test_ids=["img_02", "img_03"])


if __name__ == "__main__":
    unittest.main()
