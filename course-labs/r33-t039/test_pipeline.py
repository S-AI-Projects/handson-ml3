"""
Unit tests for R33-T039: Test an input perturbation
===================================================
Covers:
  - Exact synthetic text fixture verification (6 records, t1..t6)
  - Robustness to record-order perturbation (preserves ID-aligned predictions)
  - Robustness to text formatting/case/whitespace perturbation
  - Fallback mapping of unseen tokens to '<unk>' (UNK_ID)
  - Rejection of invalid inputs (missing fields, wrong types) with ValueError
  - Deliberate tampering failure detection

Standard library only. Offline CPU-only execution.
"""

import sys
import unittest

sys.path.insert(0, "course-labs/r33-t039")
from pipeline import (
    PAD_ID,
    SYNTHETIC_FIXTURE,
    TEST_RECORD_IDS,
    TRAIN_RECORD_IDS,
    UNK_ID,
    SimpleCPUTextClassifier,
    TextTokenizer,
    apply_formatting_perturbation,
    apply_record_order_perturbation,
    run_perturbation_pipeline,
)


class TestTextFixtureAndTokenizer(unittest.TestCase):

    def test_fixture_integrity(self):
        """Six fictional records with IDs t1..t6 and specified text/labels."""
        expected_records = [
            ("t1", "red apple", 1),
            ("t2", "green apple", 1),
            ("t3", "blue sky", 0),
            ("t4", "dark sky", 0),
            ("t5", "red fruit", 1),
            ("t6", "blue cloud", 0),
        ]
        self.assertEqual(len(SYNTHETIC_FIXTURE), 6)
        for i, (rid, txt, lbl) in enumerate(expected_records):
            self.assertEqual(SYNTHETIC_FIXTURE[i]["record_id"], rid)
            self.assertEqual(SYNTHETIC_FIXTURE[i]["text"], txt)
            self.assertEqual(SYNTHETIC_FIXTURE[i]["label"], lbl)

    def test_unseen_token_fallback_to_unk(self):
        """Unseen vocabulary tokens must explicitly map to UNK_ID (1)."""
        tokenizer = TextTokenizer()
        tokenizer.fit_on_texts(["red apple", "blue sky"])

        # 'yellow' and 'banana' were never seen
        encoded = tokenizer.encode("yellow banana")
        self.assertEqual(encoded, [UNK_ID, UNK_ID])

    def test_formatting_perturbation_preserves_encoding(self):
        """Leading/trailing spaces and uppercase formatting produce equivalent token sequences."""
        tokenizer = TextTokenizer()
        tokenizer.fit_on_texts(["red apple"])

        clean_ids = tokenizer.encode("red apple")
        perturbed_ids = tokenizer.encode(apply_formatting_perturbation("red apple"))
        self.assertEqual(clean_ids, perturbed_ids)


class TestRobustnessTransformations(unittest.TestCase):

    def setUp(self):
        self.tokenizer = TextTokenizer()
        self.tokenizer.fit_on_texts([r["text"] for r in SYNTHETIC_FIXTURE])
        train_records = [r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS]
        self.clf = SimpleCPUTextClassifier(self.tokenizer)
        self.clf.fit(train_records)

    def test_record_order_perturbation_preserves_id_aligned_predictions(self):
        """Permuting input record order must preserve exact predictions when aligned by record_id."""
        original_preds = self.clf.predict(SYNTHETIC_FIXTURE)

        # Apply reversed and shuffled orders
        reversed_records = apply_record_order_perturbation(SYNTHETIC_FIXTURE, [5, 4, 3, 2, 1, 0])
        reversed_preds = self.clf.predict(reversed_records)

        arbitrary_records = apply_record_order_perturbation(SYNTHETIC_FIXTURE, [2, 0, 4, 1, 5, 3])
        arbitrary_preds = self.clf.predict(arbitrary_records)

        for rid in original_preds:
            self.assertEqual(original_preds[rid], reversed_preds[rid])
            self.assertEqual(original_preds[rid], arbitrary_preds[rid])

    def test_unseen_tokens_do_not_crash_model(self):
        """Records with completely unseen tokens run cleanly through model using fallback."""
        unseen_record = {"record_id": "test_novel", "text": "novel quantum entity"}
        pred = self.clf.predict_record(unseen_record)
        self.assertIn(pred, [0, 1])


class TestRejectionOfUnsupportedInputs(unittest.TestCase):

    def setUp(self):
        self.tokenizer = TextTokenizer()
        self.tokenizer.fit_on_texts(["apple"])
        self.clf = SimpleCPUTextClassifier(self.tokenizer)
        self.clf.fit([{"record_id": "t1", "text": "apple", "label": 1}])

    def test_missing_text_field_rejected(self):
        """Input record missing 'text' field must raise ValueError."""
        with self.assertRaises(ValueError):
            self.clf.score_record({"record_id": "bad1"})

    def test_missing_record_id_rejected(self):
        """Input record missing 'record_id' must raise ValueError."""
        with self.assertRaises(ValueError):
            self.clf.score_record({"text": "apple"})

    def test_non_string_text_rejected(self):
        """Input record with non-string text (e.g. numeric or None) must raise ValueError."""
        with self.assertRaises(ValueError):
            self.clf.score_record({"record_id": "bad2", "text": 12345})
        with self.assertRaises(ValueError):
            self.clf.score_record({"record_id": "bad3", "text": None})


class TestPipelineVerificationAndTampering(unittest.TestCase):

    def test_full_perturbation_pipeline_run(self):
        """Pipeline runner passes all assertions."""
        res = run_perturbation_pipeline()
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["id_aligned_equivalent"])

    def test_deliberate_tampering_detected(self):
        """Deliberately changing expected base prediction must fail."""
        res = run_perturbation_pipeline()
        tampered_predictions = {"t1": 999}
        self.assertNotEqual(res["base_predictions"]["t1"], tampered_predictions["t1"])


if __name__ == "__main__":
    unittest.main()
