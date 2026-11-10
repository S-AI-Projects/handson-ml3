import json
import unittest

from fixture import RECORDS, SCHEMA, SPLITS, get_split
from report import build_report, classify, fit_linear, predict, render_text

INJECTED = {"r6": 9.0}          # r6 should be 11.0, so this is deliberately wrong
EXPECTED_ERROR_ID = "r6"


def make(injected=None):
    model = fit_linear(get_split("train"))
    return build_report(RECORDS, SPLITS, model, injected)


class TestFixture(unittest.TestCase):
    def test_formula_and_schema(self):
        self.assertEqual([r["record_id"] for r in RECORDS], ["r1", "r2", "r3", "r4", "r5", "r6"])
        self.assertEqual([r["x1"] for r in RECORDS], [0, 1, 2, 3, 4, 5])
        for r in RECORDS:
            self.assertEqual(tuple(r.keys()), SCHEMA)
            self.assertEqual(r["x2"], 1)
            self.assertEqual(r["target"], 2 * r["x1"] + 1)

    def test_splits_explicit_and_disjoint(self):
        self.assertEqual(SPLITS, {"train": ["r1", "r2", "r3", "r4"], "val": ["r5"], "test": ["r6"]})
        ids = [i for s in SPLITS.values() for i in s]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), sorted(r["record_id"] for r in RECORDS))


class TestReport(unittest.TestCase):
    def test_fit_uses_training_data_only(self):
        altered = [dict(r, target=r["target"] + 100) if r["record_id"] in ("r5", "r6") else r
                   for r in RECORDS]
        train = [r for r in altered if r["record_id"] in SPLITS["train"]]
        self.assertEqual(fit_linear(train), fit_linear(get_split("train")))
        model = fit_linear(get_split("train"))
        self.assertEqual((model["slope"], model["intercept"]), (2.0, 1.0))

    def test_clean_run_has_no_errors(self):
        s = make()["summary"]
        self.assertEqual((s["n_total"], s["n_correct"], s["n_errors"]), (6, 6, 0))
        self.assertEqual(s["error_record_ids"], [])
        self.assertEqual(s["mae"], 0.0)

    def test_injected_wrong_prediction_reported_with_correct_id(self):
        rep = make(INJECTED)
        self.assertEqual(rep["summary"]["error_record_ids"], [EXPECTED_ERROR_ID])
        row = next(x for x in rep["rows"] if x["record_id"] == "r6")
        self.assertEqual((row["split"], row["expected"], row["predicted"]), ("test", 11.0, 9.0))
        self.assertEqual((row["error"], row["abs_error"]), (-2.0, 2.0))
        self.assertEqual(row["reason_category"], "injected_error")

    def test_correct_predictions_not_in_error_list(self):
        rep = make(INJECTED)
        for x in rep["rows"]:
            if x["record_id"] != "r6":
                self.assertEqual(x["reason_category"], "correct")
                self.assertNotIn(x["record_id"], rep["summary"]["error_record_ids"])

    def test_aggregates_match_detail_rows(self):
        rep = make(INJECTED)
        rows, s = rep["rows"], rep["summary"]
        self.assertEqual(s["n_total"], len(rows))
        self.assertEqual(s["n_errors"], sum(1 for x in rows if x["reason_category"] != "correct"))
        self.assertEqual(s["n_correct"] + s["n_errors"], s["n_total"])
        self.assertAlmostEqual(s["mae"], sum(x["abs_error"] for x in rows) / len(rows))
        self.assertEqual(sum(v["n"] for v in s["by_split"].values()), s["n_total"])
        self.assertEqual(sum(v["n_errors"] for v in s["by_split"].values()), s["n_errors"])
        self.assertEqual(sum(s["by_reason"].values()), s["n_total"])
        self.assertEqual(s["by_split"]["test"], {"n": 1, "n_errors": 1})

    def test_classify_reason_categories(self):
        self.assertEqual(classify(5.0, 5.0), "correct")
        self.assertEqual(classify(5.0, 6.0), "over_prediction")
        self.assertEqual(classify(5.0, 4.0), "under_prediction")

    def test_unknown_injected_id_rejected(self):
        with self.assertRaises(ValueError):
            make({"r99": 1.0})

    def test_non_finite_injected_value_rejected(self):
        with self.assertRaises(ValueError):
            make({"r6": float("nan")})
        with self.assertRaises(ValueError):
            make({"r6": True})

    def test_report_json_roundtrip(self):
        rep = make(INJECTED)
        self.assertEqual(json.loads(json.dumps(rep)), rep)

    def test_render_text_mentions_error_record(self):
        text = render_text(make(INJECTED))
        self.assertIn("r6", text)
        self.assertIn("injected_error", text)
        self.assertIn("errors=1", text)

    def test_empty_train_rejected(self):
        with self.assertRaises(ValueError):
            fit_linear([])


if __name__ == "__main__":
    unittest.main()
