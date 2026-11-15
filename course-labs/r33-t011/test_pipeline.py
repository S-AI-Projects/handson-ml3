"""Unit tests for the R33-T011 data contract."""

import json
import math
import unittest
from pathlib import Path

from pipeline import (
    validate_records,
    validate_fixture_splits,
    format_errors,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def load_fixture():
    with FIXTURE_PATH.open(encoding="utf-8") as file:
        return json.load(file)


class TestDataContract(unittest.TestCase):

    def setUp(self):
        self.fixture = load_fixture()
        self.records = self.fixture["records"]

    def test_valid_fixture_passes(self):
        self.assertEqual(validate_records(self.records), [])

    def test_fixture_has_six_records(self):
        self.assertEqual(len(self.records), 6)

    def test_formula_is_correct(self):
        for record in self.records:
            self.assertEqual(
                record["target"],
                2 * record["x1"] + 1
            )

    def test_required_splits_are_correct(self):
        self.assertEqual(validate_fixture_splits(self.fixture), [])

    def test_missing_record_id_fails(self):
        record = {"x1": 1, "x2": 1, "target": 3}
        errors = validate_records([record])

        self.assertTrue(any(
            error.field == "record_id"
            and "missing" in error.reason.lower()
            for error in errors
        ))

    def test_empty_record_id_fails(self):
        record = {
            "record_id": " ",
            "x1": 1,
            "x2": 1,
            "target": 3
        }
        errors = validate_records([record])

        self.assertTrue(any(error.field == "record_id" for error in errors))

    def test_duplicate_record_id_fails(self):
        record = {
            "record_id": "r1",
            "x1": 1,
            "x2": 1,
            "target": 3
        }
        errors = validate_records([record, record.copy()])

        self.assertTrue(any(
            error.record_id == "r1"
            and error.field == "record_id"
            and "duplicate" in error.reason.lower()
            for error in errors
        ))

    def test_missing_numeric_field_fails(self):
        record = {
            "record_id": "r7",
            "x1": 1,
            "target": 3
        }
        errors = validate_records([record])

        self.assertTrue(any(error.field == "x2" for error in errors))

    def test_text_in_numeric_field_fails(self):
        record = {
            "record_id": "r7",
            "x1": "hello",
            "x2": 1,
            "target": 3
        }
        errors = validate_records([record])

        self.assertTrue(any(error.field == "x1" for error in errors))

    def test_boolean_in_numeric_field_fails(self):
        record = {
            "record_id": "r7",
            "x1": True,
            "x2": 1,
            "target": 3
        }
        errors = validate_records([record])

        self.assertTrue(any(error.field == "x1" for error in errors))

    def test_nan_fails(self):
        record = {
            "record_id": "r7",
            "x1": math.nan,
            "x2": 1,
            "target": 3
        }
        errors = validate_records([record])

        self.assertTrue(any(
            error.field == "x1" and "finite" in error.reason.lower()
            for error in errors
        ))

    def test_infinity_fails(self):
        for value in (math.inf, -math.inf):
            with self.subTest(value=value):
                record = {
                    "record_id": "r7",
                    "x1": value,
                    "x2": 1,
                    "target": 3
                }
                errors = validate_records([record])

                self.assertTrue(any(
                    error.field == "x1"
                    and "finite" in error.reason.lower()
                    for error in errors
                ))

    def test_nested_value_fails(self):
        record = {
            "record_id": "r7",
            "x1": [1, 2],
            "x2": 1,
            "target": 3
        }
        errors = validate_records([record])

        self.assertTrue(any(error.field == "x1" for error in errors))

    def test_malformed_record_fails(self):
        errors = validate_records([None])

        self.assertTrue(any(error.field == "<record>" for error in errors))

    def test_non_list_dataset_fails(self):
        errors = validate_records({"record_id": "r1"})

        self.assertTrue(any(error.field == "<records>" for error in errors))

    def test_unexpected_field_fails(self):
        record = {
            "record_id": "r7",
            "x1": 1,
            "x2": 1,
            "target": 3,
            "extra": "not allowed"
        }
        errors = validate_records([record])

        self.assertTrue(any(error.field == "extra" for error in errors))

    def test_error_report_has_record_id_field_and_reason(self):
        record = {
            "record_id": "r7",
            "x1": "wrong",
            "x2": 1,
            "target": 3
        }
        report = format_errors(validate_records([record]))

        self.assertTrue(len(report) > 0)
        self.assertEqual(report[0]["record_id"], "r7")
        self.assertEqual(report[0]["field"], "x1")
        self.assertTrue(report[0]["reason"])

    def test_wrong_split_fails(self):
        fixture = load_fixture()
        fixture["splits"]["test"] = ["r5", "r6"]

        errors = validate_fixture_splits(fixture)

        self.assertTrue(any(error.field == "splits" for error in errors))

    def test_overlapping_splits_fail(self):
        fixture = load_fixture()
        fixture["splits"]["test"] = ["r5", "r6"]

        errors = validate_fixture_splits(fixture)

        self.assertTrue(any(
            "more than one split" in error.reason.lower()
            for error in errors
        ))

    def test_malformed_splits_fail(self):
        fixture = load_fixture()
        fixture["splits"]["train"] = "r1"

        errors = validate_fixture_splits(fixture)

        self.assertTrue(any(error.field == "train" for error in errors))


if __name__ == "__main__":
    unittest.main()
