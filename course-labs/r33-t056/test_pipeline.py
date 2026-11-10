import json
import os
import pickle
import tempfile
import unittest

import pipeline
from fixture import RECORDS, SCHEMA, SPLITS, get_split
from pipeline import PipelineSchemaError, fit, load, predict, save

TOL = 1e-9
EXPECTED_SCHEMA_VERSION = 1


class TestSavedPipeline(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "model.json")
        self.model = fit(get_split("train"))
        save(self.model, self.path)

    def _doc(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write_bad(self, doc):
        p = os.path.join(self.dir, "bad.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        return p

    def test_before_after_predictions_agree_within_tolerance(self):
        loaded = load(self.path)
        for r in RECORDS:
            before = predict(self.model, r["pixels"])
            after = predict(loaded, r["pixels"])
            self.assertEqual(before["label"], after["label"])
            for k in before["scores"]:
                self.assertLessEqual(abs(before["scores"][k] - after["scores"][k]), TOL)

    def test_heldout_labels_match(self):
        loaded = load(self.path)
        for r in get_split("test"):
            self.assertEqual(predict(loaded, r["pixels"])["label"], r["label"])

    def test_inference_needs_no_training_data(self):
        loaded = load(self.path)
        self.assertEqual(set(loaded), {"schema_version", "model_type", "config", "params"})
        diag = [[255 if i == j else 0 for j in range(4)] for i in range(4)]
        self.assertEqual(predict(loaded, diag)["label"], 1)

    def test_saved_file_schema_version(self):
        self.assertEqual(self._doc()["schema_version"], EXPECTED_SCHEMA_VERSION)

    def test_version_mismatch_fails_clearly(self):
        doc = self._doc()
        doc["schema_version"] = 99
        with self.assertRaisesRegex(PipelineSchemaError, "schema_version"):
            load(self._write_bad(doc))

    def test_model_type_mismatch_fails_clearly(self):
        doc = self._doc()
        doc["model_type"] = "svm"
        with self.assertRaisesRegex(PipelineSchemaError, "model_type"):
            load(self._write_bad(doc))

    def test_missing_key_fails_clearly(self):
        doc = self._doc()
        del doc["params"]
        with self.assertRaisesRegex(PipelineSchemaError, "missing key"):
            load(self._write_bad(doc))

    def test_wrong_centroid_length_fails(self):
        doc = self._doc()
        doc["params"]["centroids"]["0"] = [0.0] * 15
        with self.assertRaisesRegex(PipelineSchemaError, "16 values"):
            load(self._write_bad(doc))

    def test_non_finite_value_fails(self):
        doc = self._doc()
        doc["params"]["centroids"]["1"][0] = float("nan")
        with self.assertRaises(PipelineSchemaError):
            load(self._write_bad(doc))

    def test_invalid_json_fails(self):
        p = os.path.join(self.dir, "garbage.json")
        with open(p, "w") as f:
            f.write("not json at all")
        with self.assertRaises(PipelineSchemaError):
            load(p)

    def test_pickle_file_is_rejected_not_loaded(self):
        p = os.path.join(self.dir, "model.pkl")
        with open(p, "wb") as f:
            pickle.dump(self.model, f)
        with self.assertRaises(PipelineSchemaError):
            load(p)

    def test_module_does_not_use_pickle(self):
        with open(pipeline.__file__, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("import pickle", src)
        self.assertNotIn("pickle.load", src)

    def test_predict_rejects_bad_shape(self):
        with self.assertRaises(PipelineSchemaError):
            predict(self.model, [[0, 0, 0, 0]] * 3)

    def test_predict_rejects_out_of_range_pixel(self):
        bad = [[0] * 4 for _ in range(4)]
        bad[0][0] = 300
        with self.assertRaises(PipelineSchemaError):
            predict(self.model, bad)

    def test_fit_is_deterministic(self):
        self.assertEqual(fit(get_split("train")), fit(get_split("train")))


class TestFixture(unittest.TestCase):
    def test_schema_and_size(self):
        self.assertEqual(len(RECORDS), 6)
        for r in RECORDS:
            self.assertEqual(tuple(r.keys()), SCHEMA)

    def test_unique_ids_and_labels(self):
        ids = [r["record_id"] for r in RECORDS]
        self.assertEqual(len(ids), len(set(ids)))
        labels = [r["label"] for r in RECORDS]
        self.assertEqual((labels.count(0), labels.count(1)), (3, 3))

    def test_splits_disjoint_by_group(self):
        seen = [g for gs in SPLITS.values() for g in gs]
        self.assertEqual(len(seen), len(set(seen)))
        ids = [r["record_id"] for s in SPLITS for r in get_split(s)]
        self.assertEqual(sorted(ids), sorted(r["record_id"] for r in RECORDS))


if __name__ == "__main__":
    unittest.main()
