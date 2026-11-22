"""
Unit tests for Course Lab R33-T042 deterministic split manifest and centroid assignment.
"""
import unittest
import json
import os
from pipeline import assign_nearest_centroids, generate_split_manifest, validate_manifest, CENTROIDS

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestDeterministicSplitManifest(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)

    def test_nearest_centroid_assignments_match_specification(self):
        # Six points (0,0),(0,1),(1,0),(5,5),(5,6),(6,5) with centroids (0,0), (5,5)
        # Expected assignments: 0, 0, 0, 1, 1, 1
        expected = [0, 0, 0, 1, 1, 1]
        actual = assign_nearest_centroids(self.fixture, CENTROIDS)
        self.assertEqual(actual, expected)

    def test_fixture_points_and_ids(self):
        self.assertEqual(len(self.fixture), 6)
        expected_ids = ["p1", "p2", "p3", "p4", "p5", "p6"]
        self.assertEqual([r["record_id"] for r in self.fixture], expected_ids)

    def test_split_sets_are_disjoint(self):
        manifest = generate_split_manifest(self.fixture)
        train_set = set(manifest["splits"]["train"])
        val_set = set(manifest["splits"]["validation"])
        test_set = set(manifest["splits"]["test"])

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_repeated_runs_produce_identical_manifest(self):
        m1 = generate_split_manifest(self.fixture)
        m2 = generate_split_manifest(self.fixture)
        self.assertEqual(m1, m2)

    def test_reject_overlapping_record_ids(self):
        bad_manifest = {
            "splits": {
                "train": ["p1", "p2", "p3"],
                "validation": ["p3", "p4"],
                "test": ["p5", "p6"]
            }
        }
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(bad_manifest)
        self.assertIn("overlap", str(ctx.exception).lower())

    def test_reject_overlapping_groups_when_strict(self):
        bad_manifest = {
            "splits": {
                "train": ["p1", "p2"],   # group cluster_0
                "validation": ["p4"],     # group cluster_1
                "test": ["p3"]            # group cluster_0 -> group overlap with train!
            }
        }
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(bad_manifest, records=self.fixture, strict_group_isolation=True)
        self.assertIn("groups", str(ctx.exception).lower())

    def test_deliberate_assignment_change_fails(self):
        # Alter point p4 to (0, 0) and verify assignment changes
        modified = json.loads(json.dumps(self.fixture))
        modified[3]["x"] = 0.0
        modified[3]["y"] = 0.0
        actual = assign_nearest_centroids(modified)
        self.assertNotEqual(actual, [0, 0, 0, 1, 1, 1])

if __name__ == "__main__":
    unittest.main()
