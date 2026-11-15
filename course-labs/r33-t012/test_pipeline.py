import unittest
from pathlib import Path

from pipeline import (
    create_manifest,
    load_fixture,
    validate_manifest,
    validate_records,
)


FIXTURE = Path(__file__).with_name("fixture.json")


class TestClassificationSplit(unittest.TestCase):

    def setUp(self):
        self.records = load_fixture(FIXTURE)

    def test_fixture_has_eight_records(self):
        self.assertEqual(len(self.records), 8)

    def test_exact_split_manifest(self):
        expected = {
            "train": ["r2", "r3", "r7", "r8"],
            "validation": ["r4", "r6"],
            "test": ["r1", "r5"],
        }
        self.assertEqual(create_manifest(self.records), expected)

    def test_repeated_runs_are_deterministic(self):
        first = create_manifest(self.records, seed=42)
        second = create_manifest(self.records, seed=42)
        self.assertEqual(first, second)

    def test_record_ids_are_disjoint(self):
        manifest = create_manifest(self.records)
        train = set(manifest["train"])
        validation = set(manifest["validation"])
        test = set(manifest["test"])

        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))

    def test_groups_are_disjoint(self):
        manifest = create_manifest(self.records)
        groups = {
            record["record_id"]: record["group_id"]
            for record in self.records
        }

        split_groups = [
            {groups[record_id] for record_id in manifest[split]}
            for split in ("train", "validation", "test")
        ]

        self.assertTrue(split_groups[0].isdisjoint(split_groups[1]))
        self.assertTrue(split_groups[0].isdisjoint(split_groups[2]))
        self.assertTrue(split_groups[1].isdisjoint(split_groups[2]))

    def test_duplicate_record_id_is_rejected(self):
        bad_records = self.records + [dict(self.records[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
            validate_records(bad_records)

    def test_missing_field_is_rejected(self):
        bad_records = [dict(record) for record in self.records]
        del bad_records[0]["group_id"]

        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_records(bad_records)

    def test_non_numeric_feature_is_rejected(self):
        bad_records = [dict(record) for record in self.records]
        bad_records[0]["x1"] = "not-a-number"

        with self.assertRaisesRegex(ValueError, "must be numeric"):
            validate_records(bad_records)

    def test_invalid_label_is_rejected(self):
        bad_records = [dict(record) for record in self.records]
        bad_records[0]["label"] = 3

        with self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
            validate_records(bad_records)

    def test_group_leakage_is_rejected(self):
        bad_records = [dict(record) for record in self.records]
        bad_records[1]["group_id"] = bad_records[0]["group_id"]

        manifest = {
            "train": ["r1"],
            "validation": ["r2"],
            "test": ["r3", "r4", "r5", "r6", "r7", "r8"],
        }

        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_manifest(bad_records, manifest)

    def test_missing_manifest_record_is_rejected(self):
        manifest = {
            "train": ["r1", "r2", "r5", "r6"],
            "validation": ["r3"],
            "test": ["r4", "r8"],
        }

        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            validate_manifest(self.records, manifest)


if __name__ == "__main__":
    unittest.main()
