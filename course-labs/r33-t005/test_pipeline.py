"""
Unit test suite for Course Lab R33-T005 preprocessing contract.
Tests parameter isolation, shape/dtype retention, contamination rejection,
and transform immutability.
"""

from pathlib import Path
import math
import unittest

from pipeline import (
    ContaminationError,
    ExplicitPreprocessor,
    NotFittedError,
    SchemaError,
    load_dataset,
)


class TestR33T005PreprocessingContract(unittest.TestCase):
    """Offline, CPU-only unit tests for preprocessing parameter isolation."""

    def setUp(self) -> None:
        self.lab_dir = Path(__file__).parent
        self.json_path = self.lab_dir / "fixture.json"
        self.csv_path = self.lab_dir / "fixture.csv"
        self.fixture = load_dataset(self.json_path)
        self.train_records = [r for r in self.fixture if r.get("split") == "train"]
        self.val_records = [r for r in self.fixture if r.get("split") == "validation"]
        self.test_records = [r for r in self.fixture if r.get("split") == "test"]

    # 1. Exact Synthetic Formula Fixture
    def test_fixture_dimensions_formula_and_splits(self) -> None:
        """Fixture must contain 6 rows: r1..r4 (train), r5 (val), r6 (test) obeying target=2*x1+1."""
        self.assertEqual(len(self.fixture), 6)
        self.assertEqual(len(self.train_records), 4)
        self.assertEqual(len(self.val_records), 1)
        self.assertEqual(len(self.test_records), 1)

        expected_x1 = [0, 1, 2, 3, 4, 5]
        expected_x2 = [1, 1, 1, 1, 1, 1]
        expected_targets = [1, 3, 5, 7, 9, 11]

        actual_x1 = [r["x1"] for r in self.fixture]
        actual_x2 = [r["x2"] for r in self.fixture]
        actual_targets = [r["target"] for r in self.fixture]

        self.assertEqual(actual_x1, expected_x1)
        self.assertEqual(actual_x2, expected_x2)
        self.assertEqual(actual_targets, expected_targets)

    def test_csv_fixture_loading_parity(self) -> None:
        """CSV fixture must load identically to JSON fixture."""
        csv_records = load_dataset(self.csv_path)
        self.assertEqual(csv_records, self.fixture)

    # 2. Output Values, Dtype, and Shape
    def test_output_values_dtype_and_shape_on_training_fit(self) -> None:
        """
        Fitting on train (r1..r4, x1=[0,1,2,3]) yields mean=1.5, var=1.25, std=sqrt(1.25).
        Transformed rows must have exact shape (2 features), float dtype, and asserted values.
        """
        preprocessor = ExplicitPreprocessor(feature_order=("x1", "x2"))
        preprocessor.fit(self.train_records)

        expected_mean_x1 = 1.5
        expected_std_x1 = math.sqrt(1.25)
        self.assertAlmostEqual(preprocessor.state_.means["x1"], expected_mean_x1, places=7)
        self.assertAlmostEqual(preprocessor.state_.stds["x1"], expected_std_x1, places=7)
        self.assertEqual(preprocessor.state_.means["x2"], 1.0)
        self.assertEqual(preprocessor.state_.stds["x2"], 1.0)  # constant scale fallback

        # Transform training rows
        train_out = preprocessor.transform(self.train_records)
        self.assertEqual(len(train_out), 4)

        for row in train_out:
            vec = row["_feature_vector"]
            self.assertEqual(len(vec), 2, "Feature vector shape must be exactly 2")
            self.assertIsInstance(vec[0], float)
            self.assertIsInstance(vec[1], float)
            self.assertEqual(row["x2"], 0.0)  # (1 - 1) / 1.0 = 0.0

        # Assert standardized x1 values
        self.assertAlmostEqual(train_out[0]["x1"], (0 - 1.5) / expected_std_x1, places=7)
        self.assertAlmostEqual(train_out[1]["x1"], (1 - 1.5) / expected_std_x1, places=7)
        self.assertAlmostEqual(train_out[2]["x1"], (2 - 1.5) / expected_std_x1, places=7)
        self.assertAlmostEqual(train_out[3]["x1"], (3 - 1.5) / expected_std_x1, places=7)

        # Transform held-out validation and test rows using ONLY training statistics
        val_out = preprocessor.transform(self.val_records)
        self.assertAlmostEqual(val_out[0]["x1"], (4 - 1.5) / expected_std_x1, places=7)

        test_out = preprocessor.transform(self.test_records)
        self.assertAlmostEqual(test_out[0]["x1"], (5 - 1.5) / expected_std_x1, places=7)

    # 3. Parameter Immutability via Transform
    def test_transform_is_read_only_and_cannot_fit(self) -> None:
        """Calling transform on held-out data must NOT alter any learned parameters."""
        preprocessor = ExplicitPreprocessor()
        preprocessor.fit(self.train_records)

        saved_means = dict(preprocessor.state_.means)
        saved_stds = dict(preprocessor.state_.stds)

        # Transform test data (x1=5)
        _ = preprocessor.transform(self.test_records)

        # Ensure parameters remained bitwise identical
        self.assertEqual(preprocessor.state_.means, saved_means)
        self.assertEqual(preprocessor.state_.stds, saved_stds)

    def test_transform_before_fit_raises_error(self) -> None:
        """Transforming before fit must raise NotFittedError."""
        preprocessor = ExplicitPreprocessor()
        with self.assertRaises(NotFittedError):
            preprocessor.transform(self.train_records)

    # 4. Rejection of Held-Out Data in fit()
    def test_fitting_on_held_out_data_rejected(self) -> None:
        """Supplying validation or test records to fit() must raise ContaminationError."""
        preprocessor = ExplicitPreprocessor()
        with self.assertRaises(ContaminationError):
            preprocessor.fit(self.fixture)  # Contains r5 (val) and r6 (test)

        with self.assertRaises(ContaminationError):
            preprocessor.fit(self.val_records)

    # 5. Rejection of Deliberate Contamination Fixtures
    def test_deliberate_contamination_fixture_rejected(self) -> None:
        """Records marked with contamination or infinite values must be rejected."""
        preprocessor = ExplicitPreprocessor()

        contaminated_record = [
            {"record_id": "c1", "x1": 0, "x2": 1, "target": 1, "split": "train", "_contaminated": True}
        ]
        with self.assertRaises(ContaminationError):
            preprocessor.fit(contaminated_record)

        infinite_record = [
            {"record_id": "c2", "x1": float("inf"), "x2": 1, "target": 1, "split": "train"}
        ]
        with self.assertRaises(ContaminationError):
            preprocessor.fit(infinite_record)

    # 6. Missing Value Imputation
    def test_missing_value_imputed_with_training_mean(self) -> None:
        """A missing feature value in test data is imputed with training mean and scales to 0.0."""
        preprocessor = ExplicitPreprocessor()
        preprocessor.fit(self.train_records)

        record_with_nan = [{"record_id": "test_nan", "x1": None, "x2": 1, "target": 1}]
        transformed = preprocessor.transform(record_with_nan)

        # Imputed value is 1.5, (1.5 - 1.5) / std = 0.0
        self.assertEqual(transformed[0]["x1"], 0.0)


if __name__ == "__main__":
    unittest.main()
