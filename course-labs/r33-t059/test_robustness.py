import copy
import os
import tempfile
import unittest

import fixture
import pipeline
from pipeline import SchemaError

SEED = 59
EXPECTED = {
    "predictions": {"r0": 0, "r1": 1, "r2": 0, "r3": 1, "r4": 0, "r5": 1},
    "accuracy": 1.0,
    "class_counts": {0: 3, 1: 3},
    "fallback_ids": [],
}
# Set by check.py to prove a changed expectation makes the suite fail.
if os.environ.get("R33_T059_MUTATE_EXPECTED"):
    EXPECTED = copy.deepcopy(EXPECTED)
    EXPECTED["accuracy"] = 0.5


class FixtureContract(unittest.TestCase):
    def test_fixture_shape_and_ids(self):
        recs = fixture.FIXTURE
        self.assertEqual(len(recs), 6)
        self.assertEqual(len({r["record_id"] for r in recs}), 6)
        self.assertEqual(sorted({r["group_id"] for r in recs}), ["g0", "g1", "g2"])
        zeros = [r for r in recs if r["label"] == 0]
        diags = [r for r in recs if r["label"] == 1]
        self.assertEqual(len(zeros), 3)
        self.assertEqual(len(diags), 3)
        for r in zeros:
            self.assertEqual(r["pixels"], [0] * 16)
        for r in diags:
            self.assertEqual(sum(1 for p in r["pixels"] if p == 255), 4)
            self.assertTrue(all(r["pixels"][i * 4 + i] == 255 for i in range(4)))
            self.assertEqual(sum(1 for p in r["pixels"] if p not in (0, 255)), 0)


class Codec(unittest.TestCase):
    def test_crc32c_known_answer(self):
        from tfrecord_io import crc32c
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)  # standard check value


class Robustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _rt(self, name, recs):
        return pipeline.roundtrip(os.path.join(self.tmp, name), recs)

    def test_baseline_result_matches_documented(self):
        got = pipeline.summarise(self._rt("a.tfrecord.gz", fixture.FIXTURE))
        self.assertEqual(got, EXPECTED)

    def test_record_order_change_preserved_after_id_alignment(self):
        shuf = pipeline.shuffled(fixture.FIXTURE, SEED)
        self.assertNotEqual([r["record_id"] for r in shuf],
                            [r["record_id"] for r in fixture.FIXTURE],
                            "seed must actually change record order")
        got = pipeline.summarise(self._rt("b.tfrecord.gz", shuf))
        self.assertEqual(got, EXPECTED)

    def test_gzip_roundtrip_is_lossless(self):
        raw = self._rt("c.tfrecord.gz", fixture.FIXTURE)
        self.assertEqual(len(raw), 6)
        self.assertEqual(raw[1]["pixels"], fixture.FIXTURE[1]["pixels"])
        self.assertEqual(raw[1]["record_id"].decode(), "r1")

    def test_corrupt_file_fails_crc(self):
        path = os.path.join(self.tmp, "d.tfrecord.gz")
        import gzip
        pipeline.roundtrip(path, fixture.FIXTURE)
        with gzip.open(path, "rb") as fh:
            blob = bytearray(fh.read())
        blob[20] ^= 0xFF
        with gzip.open(path, "wb") as fh:
            fh.write(bytes(blob))
        with self.assertRaises(ValueError):
            pipeline.read_tfrecord_gz(path)

    def test_missing_field_rejected(self):
        recs = copy.deepcopy(fixture.FIXTURE)
        raw = self._rt("e.tfrecord.gz", recs)
        del raw[2]["group_id"]
        with self.assertRaisesRegex(SchemaError, "missing field.*group_id"):
            pipeline.summarise(raw)

    def test_shape_mismatch_rejected(self):
        recs = copy.deepcopy(fixture.FIXTURE)
        recs[3]["pixels"] = recs[3]["pixels"][:9]  # 3x3 instead of 4x4
        raw = self._rt("f.tfrecord.gz", recs)
        with self.assertRaisesRegex(SchemaError, "shape mismatch"):
            pipeline.summarise(raw)

    def test_unseen_label_mapped_to_explicit_fallback(self):
        recs = copy.deepcopy(fixture.FIXTURE)
        recs[4]["label"] = 7  # unseen category
        got = pipeline.summarise(self._rt("g.tfrecord.gz", recs))
        self.assertEqual(got["fallback_ids"], ["r4"])
        self.assertEqual(got["class_counts"], {-1: 1, 0: 2, 1: 3})
        self.assertEqual(got["accuracy"], 1.0)  # fallback record is not scored

    def test_duplicate_record_id_rejected(self):
        recs = copy.deepcopy(fixture.FIXTURE)
        recs[1]["record_id"] = "r0"
        with self.assertRaisesRegex(SchemaError, "duplicate"):
            pipeline.summarise(self._rt("h.tfrecord.gz", recs))

    def test_time_order_is_never_randomised(self):
        # Image interface: only record order is permuted (pipeline.shuffled);
        # nothing here touches temporal order.
        self.assertFalse(hasattr(pipeline, "shuffle_time"))


if __name__ == "__main__":
    unittest.main()
