import json
import tempfile
import unittest
from pathlib import Path

from pipeline import (
    MODEL_VERSION,
    PipelineError,
    load_pipeline,
    predict,
    save_pipeline,
    train_pipeline,
)


HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))


class SavedPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        records = FIXTURE["records"]
        ids = FIXTURE["splits"]["train"]
        training_records = [r for r in records if r["record_id"] in ids]
        cls.model = train_pipeline(training_records)

    def test_fixture_has_six_records(self):
        self.assertEqual(len(FIXTURE["records"]), 6)

    def test_fixture_ids_and_labels(self):
        self.assertEqual(
            [r["record_id"] for r in FIXTURE["records"]],
            ["t1", "t2", "t3", "t4", "t5", "t6"],
        )
        self.assertEqual(
            [r["label"] for r in FIXTURE["records"]],
            [1, 1, 0, 0, 1, 0],
        )

    def test_splits_are_explicit_and_disjoint(self):
        splits = FIXTURE["splits"]
        self.assertEqual(splits["train"], ["t1", "t2", "t3", "t4"])
        self.assertEqual(splits["validation"], ["t5"])
        self.assertEqual(splits["test"], ["t6"])

        all_ids = splits["train"] + splits["validation"] + splits["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {r["record_id"] for r in FIXTURE["records"]})

    def test_expected_validation_prediction(self):
        result = predict(self.model, {"record_id": "t5", "text": "red fruit"})
        self.assertEqual(result["label"], 1)

    def test_expected_test_prediction(self):
        result = predict(self.model, {"record_id": "t6", "text": "blue cloud"})
        self.assertEqual(result["label"], 0)

    def test_save_and_reload_preserves_predictions(self):
        records = [
            {"record_id": "t5", "text": "red fruit"},
            {"record_id": "t6", "text": "blue cloud"},
        ]
        before = [predict(self.model, r) for r in records]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            save_pipeline(self.model, path)
            loaded = load_pipeline(path)

        after = [predict(loaded, r) for r in records]

        for first, second in zip(before, after):
            self.assertEqual(first["record_id"], second["record_id"])
            self.assertEqual(first["label"], second["label"])
            for label in first["scores"]:
                self.assertAlmostEqual(
                    first["scores"][label],
                    second["scores"][label],
                    delta=1e-12,
                )

    def test_prediction_does_not_need_training_records(self):
        result = predict(self.model, {"record_id": "new1", "text": "red apple"})
        self.assertEqual(result["label"], 1)

    def test_missing_record_id_fails(self):
        with self.assertRaisesRegex(PipelineError, "record_id"):
            predict(self.model, {"text": "red apple"})

    def test_missing_text_fails(self):
        with self.assertRaisesRegex(PipelineError, "text"):
            predict(self.model, {"record_id": "new1"})

    def test_empty_text_fails(self):
        with self.assertRaisesRegex(PipelineError, "text"):
            predict(self.model, {"record_id": "new1", "text": "  "})

    def test_wrong_record_id_type_fails(self):
        with self.assertRaisesRegex(PipelineError, "record_id"):
            predict(self.model, {"record_id": 123, "text": "red apple"})

    def test_unexpected_prediction_field_fails(self):
        with self.assertRaisesRegex(PipelineError, "unexpected field"):
            predict(
                self.model,
                {"record_id": "new1", "text": "red apple", "label": 1},
            )

    def test_unsupported_model_version_fails(self):
        bad_model = dict(self.model)
        bad_model["version"] = MODEL_VERSION + 1

        with self.assertRaisesRegex(PipelineError, "unsupported model version"):
            predict(bad_model, {"record_id": "new1", "text": "red apple"})

    def test_wrong_model_format_fails(self):
        bad_model = dict(self.model)
        bad_model["format"] = "unknown-format"

        with self.assertRaisesRegex(PipelineError, "format mismatch"):
            predict(bad_model, {"record_id": "new1", "text": "red apple"})

    def test_missing_model_parameter_fails(self):
        bad_model = dict(self.model)
        bad_model["class_log_prior"] = {}

        with self.assertRaisesRegex(PipelineError, "missing class"):
            predict(bad_model, {"record_id": "new1", "text": "red apple"})

    def test_invalid_training_label_fails(self):
        bad_records = [
            {"record_id": "a", "text": "red apple", "label": 1},
            {"record_id": "b", "text": "blue sky", "label": 2},
        ]

        with self.assertRaisesRegex(PipelineError, "label"):
            train_pipeline(bad_records)

    def test_duplicate_training_id_fails(self):
        bad_records = [
            {"record_id": "a", "text": "red apple", "label": 1},
            {"record_id": "a", "text": "blue sky", "label": 0},
        ]

        with self.assertRaisesRegex(PipelineError, "duplicate ID"):
            train_pipeline(bad_records)


if __name__ == "__main__":
    unittest.main()
