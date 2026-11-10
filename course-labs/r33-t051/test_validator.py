import unittest
import fixture
from validator import validate_records


class TestValidator(unittest.TestCase):
    def test_valid_fixture_passes(self):
        self.assertEqual(validate_records(fixture.RECORDS), [])

    def test_fixture_matches_spec(self):
        self.assertEqual([r["record_id"] for r in fixture.RECORDS], ["t1", "t2", "t3", "t4", "t5", "t6"])
        self.assertEqual([r["label"] for r in fixture.RECORDS], [1, 1, 0, 0, 1, 0])

    def test_missing_id_fails(self):
        recs = [{"text": "red apple", "label": 1}]
        f = validate_records(recs)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].field, "record_id")
        self.assertIn("missing record_id", f[0].reason)

    def test_malformed_text_fails(self):
        for bad in [123, ["red", "apple"], b"bytes", ""]:
            recs = [{"record_id": "x1", "text": bad, "label": 1}]
            f = validate_records(recs)
            self.assertEqual(len(f), 1, msg=repr(bad))
            self.assertEqual((f[0].record_id, f[0].field), ("x1", "text"))

    def test_malformed_numeric_label_fails(self):
        for bad in ["1", 1.5, True, 2, -1, [1]]:
            recs = [{"record_id": "x1", "text": "red apple", "label": bad}]
            f = validate_records(recs)
            self.assertEqual(len(f), 1, msg=repr(bad))
            self.assertEqual(f[0].field, "label")

    def test_duplicate_id_fails(self):
        recs = [
            {"record_id": "t1", "text": "red apple", "label": 1},
            {"record_id": "t1", "text": "green apple", "label": 1},
        ]
        f = validate_records(recs)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].reason, "duplicate record_id")

    def test_no_record_silently_dropped(self):
        recs = fixture.RECORDS + [{"record_id": "bad", "text": 5, "label": "x"}]
        f = validate_records(recs)
        self.assertEqual({x.record_id for x in f}, {"bad"})
        self.assertEqual(len(f), 2)

    def test_splits_are_disjoint_and_complete(self):
        s = fixture.SPLITS
        all_ids = s["train"] + s["validation"] + s["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {r["record_id"] for r in fixture.RECORDS})


if __name__ == "__main__":
    unittest.main()
