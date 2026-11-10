import copy
import json
import unittest
from pathlib import Path

import fixture
from assign import nearest_centroid
from manifest import SPLIT_NAMES, ManifestError, make_manifest, validate_manifest

PINNED_SPLITS = {
    "train": ["p2", "p3", "p4", "p5"],
    "validation": ["p6"],
    "test": ["p1"],
}


def grouped_records():
    spec = [("r1", "A"), ("r2", "A"), ("r3", "B"), ("r4", "B"), ("r5", "C"), ("r6", "D")]
    return [
        {"record_id": rid, "x1": n, "x2": 0, "group": g}
        for n, (rid, g) in enumerate(spec)
    ]


def timed_records():
    spec = [("r4", 4), ("r1", 1), ("r6", 6), ("r2", 2), ("r5", 5), ("r3", 3)]
    return [{"record_id": rid, "x1": 0, "x2": 0, "time": t} for rid, t in spec]


class TestFixture(unittest.TestCase):
    def test_fixture_matches_spec(self):
        self.assertEqual(fixture.IDS, ["p1", "p2", "p3", "p4", "p5", "p6"])
        self.assertEqual(
            [(r["x1"], r["x2"]) for r in fixture.RECORDS],
            [(0, 0), (0, 1), (1, 0), (5, 5), (5, 6), (6, 5)],
        )
        self.assertEqual(fixture.CENTROIDS, [(0, 0), (5, 5)])

    def test_nearest_centroid_assignments(self):
        result = nearest_centroid(fixture.RECORDS, fixture.CENTROIDS)
        self.assertEqual(result, [0, 0, 0, 1, 1, 1])
        self.assertEqual(result, fixture.EXPECTED_ASSIGNMENTS)


class TestManifest(unittest.TestCase):
    def test_pinned_manifest_ids(self):
        manifest = make_manifest(fixture.RECORDS, seed=fixture.SEED)
        self.assertEqual(manifest["splits"], PINNED_SPLITS)

    def test_splits_disjoint_and_cover_all(self):
        splits = make_manifest(fixture.RECORDS, seed=0)["splits"]
        all_ids = [i for name in SPLIT_NAMES for i in splits[name]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {r["record_id"] for r in fixture.RECORDS})
        for name in SPLIT_NAMES:
            self.assertTrue(splits[name], name)

    def test_repeated_runs_identical(self):
        runs = [
            json.dumps(make_manifest(fixture.RECORDS, seed=0), sort_keys=True)
            for _ in range(3)
        ]
        self.assertEqual(len(set(runs)), 1)

    def test_input_order_does_not_matter(self):
        forward = make_manifest(fixture.RECORDS, seed=0)
        backward = make_manifest(list(reversed(fixture.RECORDS)), seed=0)
        self.assertEqual(forward, backward)

    def test_committed_manifest_file_matches(self):
        path = Path(__file__).resolve().parent / "split_manifest.json"
        self.assertTrue(path.exists(), "run manifest.py first")
        on_disk = json.loads(path.read_text())
        expected = make_manifest(fixture.RECORDS, seed=fixture.SEED)
        self.assertEqual(on_disk, expected)


class TestRejections(unittest.TestCase):
    def test_overlapping_ids_rejected(self):
        manifest = copy.deepcopy(make_manifest(fixture.RECORDS, seed=0))
        manifest["splits"]["train"].append("p6")
        with self.assertRaises(ManifestError) as ctx:
            validate_manifest(manifest, fixture.RECORDS)
        self.assertIn("p6", str(ctx.exception))
        self.assertIn("both", str(ctx.exception))

    def test_missing_record_rejected(self):
        manifest = copy.deepcopy(make_manifest(fixture.RECORDS, seed=0))
        manifest["splits"]["train"].remove("p2")
        with self.assertRaises(ManifestError) as ctx:
            validate_manifest(manifest, fixture.RECORDS)
        self.assertIn("p2 is missing", str(ctx.exception))

    def test_duplicate_record_ids_in_input_rejected(self):
        records = fixture.RECORDS + [{"record_id": "p1", "x1": 9, "x2": 9}]
        with self.assertRaises(ManifestError) as ctx:
            make_manifest(records, seed=0)
        self.assertIn("duplicate record_id p1", str(ctx.exception))

    def test_missing_record_id_rejected(self):
        records = fixture.RECORDS + [{"x1": 9, "x2": 9}]
        with self.assertRaises(ManifestError) as ctx:
            make_manifest(records, seed=0)
        self.assertIn("missing or invalid record_id", str(ctx.exception))

    def test_too_few_records_rejected(self):
        with self.assertRaises(ManifestError):
            make_manifest(fixture.RECORDS[:2], seed=0)


class TestIsolation(unittest.TestCase):
    def test_group_isolation_preserved(self):
        records = grouped_records()
        splits = make_manifest(records, seed=0)["splits"]
        group_of = {r["record_id"]: r["group"] for r in records}
        for name in SPLIT_NAMES:
            self.assertTrue(splits[name], name)
        seen = {}
        for name in SPLIT_NAMES:
            for rid in splits[name]:
                seen.setdefault(group_of[rid], set()).add(name)
        for group, where in seen.items():
            self.assertEqual(len(where), 1, msg=f"group {group} in {where}")

    def test_overlapping_groups_rejected(self):
        records = grouped_records()
        manifest = {
            "splits": {
                "train": ["r1", "r3"],
                "validation": ["r2", "r5"],
                "test": ["r4", "r6"],
            }
        }
        with self.assertRaises(ManifestError) as ctx:
            validate_manifest(manifest, records)
        self.assertIn("group A", str(ctx.exception))
        self.assertIn("group B", str(ctx.exception))

    def test_shared_user_links_groups(self):
        records = [
            {"record_id": "r1", "x1": 0, "x2": 0, "group": "A", "user": "u1"},
            {"record_id": "r2", "x1": 0, "x2": 0, "group": "B", "user": "u1"},
            {"record_id": "r3", "x1": 0, "x2": 0, "group": "C", "user": "u2"},
            {"record_id": "r4", "x1": 0, "x2": 0, "group": "D", "user": "u3"},
            {"record_id": "r5", "x1": 0, "x2": 0, "group": "E", "user": "u4"},
            {"record_id": "r6", "x1": 0, "x2": 0, "group": "F", "user": "u5"},
        ]
        splits = make_manifest(records, seed=0)["splits"]
        where = {rid: name for name in SPLIT_NAMES for rid in splits[name]}
        self.assertEqual(where["r1"], where["r2"])


class TestTimeSeries(unittest.TestCase):
    def test_split_is_chronological(self):
        manifest = make_manifest(timed_records(), seed=0)
        self.assertEqual(manifest["strategy"], "chronological")
        self.assertEqual(
            manifest["splits"],
            {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"],
            },
        )

    def test_non_chronological_split_rejected(self):
        manifest = {
            "splits": {
                "train": ["r1", "r2", "r3", "r6"],
                "validation": ["r5"],
                "test": ["r4"],
            }
        }
        with self.assertRaises(ManifestError) as ctx:
            validate_manifest(manifest, timed_records())
        self.assertIn("not before", str(ctx.exception))

    def test_mixed_time_presence_rejected(self):
        records = timed_records()
        del records[0]["time"]
        with self.assertRaises(ManifestError):
            make_manifest(records, seed=0)


if __name__ == "__main__":
    unittest.main()
