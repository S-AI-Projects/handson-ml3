import os
import tempfile
import unittest

import pipeline as p


class OrderInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = p.make_fixture()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_fixture_shape_and_labels(self):
        recs = self.fixture
        self.assertEqual(len(recs), 6)
        self.assertEqual(sorted(r["label"] for r in recs), [0, 0, 0, 1, 1, 1])
        self.assertEqual(len({r["record_id"] for r in recs}), 6)
        self.assertGreater(len({r["group_id"] for r in recs}), 1)
        for r in recs:
            if r["label"] == 1:
                self.assertTrue(all(r["pixels"][d][d] == 255 for d in range(4)))
            else:
                self.assertTrue(all(v == 0 for row in r["pixels"] for v in row))

    def test_roundtrip_compressed(self):
        f = self.path("a.tfrecord.gz")
        p.write_tfrecord_gz(f, self.fixture)
        self.assertEqual(p.read_tfrecord_gz(f), self.fixture)

    def test_file_is_gzip(self):
        f = self.path("a.tfrecord.gz")
        p.write_tfrecord_gz(f, self.fixture)
        with open(f, "rb") as fh:
            self.assertEqual(fh.read(2), b"\x1f\x8b")

    def test_corrupt_crc_detected(self):
        import gzip
        f = self.path("a.tfrecord.gz")
        p.write_tfrecord_gz(f, self.fixture)
        raw = bytearray(gzip.open(f, "rb").read())
        raw[20] ^= 0xFF
        with gzip.open(f, "wb") as fh:
            fh.write(bytes(raw))
        with self.assertRaises(ValueError):
            p.read_tfrecord_gz(f)

    def test_record_order_does_not_change_result(self):
        base = p.evaluate(self.fixture)
        order = p.shuffled(self.fixture)
        self.assertNotEqual([r["record_id"] for r in order],
                            [r["record_id"] for r in self.fixture])
        f = self.path("shuf.tfrecord.gz")
        p.write_tfrecord_gz(f, order)
        shuf = p.evaluate(p.read_tfrecord_gz(f))
        self.assertEqual(base, shuf)
        self.assertEqual(base["accuracy"], 1.0)
        self.assertEqual(base["scored"], 6)

    def test_unaligned_order_would_differ(self):
        """Sanity: raw positional order differs, so alignment is what matters."""
        order = p.shuffled(self.fixture)
        self.assertNotEqual([r["record_id"] for r in order],
                            sorted(r["record_id"] for r in order))

    def test_missing_field_rejected(self):
        bad = dict(self.fixture[0])
        del bad["pixels"]
        with self.assertRaisesRegex(ValueError, "missing field: pixels"):
            p.evaluate([bad])

    def test_shape_mismatch_rejected(self):
        bad = dict(self.fixture[0])
        bad["pixels"] = [[0, 0, 0]] * 3
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            p.evaluate([bad])

    def test_unseen_label_falls_back_and_is_excluded(self):
        extra = dict(self.fixture[0])
        extra["record_id"] = "rec-99"
        extra["label"] = 7
        res = p.evaluate(self.fixture + [extra])
        self.assertIn(("rec-99", -1, 0), res["rows"])
        self.assertEqual(res["scored"], 6)
        self.assertEqual(res["accuracy"], 1.0)

    def test_seeded_shuffle_is_deterministic(self):
        a = [r["record_id"] for r in p.shuffled(self.fixture)]
        b = [r["record_id"] for r in p.shuffled(self.fixture)]
        self.assertEqual(a, b)

    def test_nonzero_test_count_guard(self):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(OrderInvarianceTests)
        self.assertGreater(suite.countTestCases(), 0)


if __name__ == "__main__":
    unittest.main()
