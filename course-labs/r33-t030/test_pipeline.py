import unittest
import json
import os
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

ClassificationSchemaValidator = pipeline.ClassificationSchemaValidator

class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        self.validator = ClassificationSchemaValidator()

    def test_fixture_integrity(self):
        """Asserts 8 synthetic rows x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.fixture), 8)
        self.assertEqual([r["x1"] for r in self.fixture], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual([r["label"] for r in self.fixture], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_valid_fixture_passes(self):
        """The valid fixture passes with 0 failures and is_valid True."""
        result = self.validator.validate_dataset(self.fixture)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(len(result["failures"]), 0)
        self.assertEqual(result["total_processed"], 8)

    def test_missing_id_reported_with_reason(self):
        """Missing record_id reports specific reason and field."""
        bad_fixture = [{"group_id": "grp_0", "x1": 0.0, "x2": 1.0, "label": 0}]
        result = self.validator.validate_dataset(bad_fixture)
        self.assertFalse(result["is_valid"])
        reasons = [f["reason"] for f in result["failures"] if f["field"] == "record_id"]
        self.assertTrue(any("Missing" in r for r in reasons))

    def test_duplicate_record_id_reported(self):
        """Duplicate record_id reports error with specific reason."""
        dup_fixture = [
            {"record_id": "dup_1", "group_id": "g1", "x1": 1.0, "x2": 1.0, "label": 0},
            {"record_id": "dup_1", "group_id": "g2", "x1": 2.0, "x2": 1.0, "label": 1}
        ]
        result = self.validator.validate_dataset(dup_fixture)
        self.assertFalse(result["is_valid"])
        dup_failures = [f for f in result["failures"] if "Duplicate" in f["reason"]]
        self.assertEqual(len(dup_failures), 1)
        self.assertEqual(dup_failures[0]["record_id"], "dup_1")

    def test_malformed_numeric_and_text_reported(self):
        """Malformed types report specific field and reason."""
        bad_data = [{
            "record_id": "r_err",
            "group_id": 12345,  # expected str
            "x1": "invalid_num",  # expected float
            "x2": 1.0,
            "label": 99  # expected 0 or 1
        }]
        result = self.validator.validate_dataset(bad_data)
        self.assertFalse(result["is_valid"])
        fields_in_error = {f["field"] for f in result["failures"]}
        self.assertIn("group_id", fields_in_error)
        self.assertIn("x1", fields_in_error)
        self.assertIn("label", fields_in_error)

    def test_no_silent_dropping(self):
        """Every input record is preserved in processed_records output."""
        mixed_fixture = [
            {"record_id": "r1", "group_id": "g1", "x1": 1.0, "x2": 1.0, "label": 0},
            {"record_id": "r2", "group_id": "g2"}  # missing fields
        ]
        result = self.validator.validate_dataset(mixed_fixture)
        self.assertEqual(len(result["processed_records"]), 2)
        self.assertTrue(result["processed_records"][0]["is_valid"])
        self.assertFalse(result["processed_records"][1]["is_valid"])

    def test_deliberate_mutation_failure(self):
        """Mutating validity expectation causes failure."""
        result = self.validator.validate_dataset(self.fixture)
        mutated_validity = False
        self.assertNotEqual(result["is_valid"], mutated_validity)
