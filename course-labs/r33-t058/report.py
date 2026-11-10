"""Error-analysis report for the synthetic regression fixture.

Model: least-squares line on x1, fitted on TRAINING records only
(x2 is the constant 1, so it carries no information and is ignored).

Each report row: record_id, split, expected, predicted, error
(predicted - expected), abs_error, reason_category.
Reason categories:
  correct          |error| <= TOL
  over_prediction  predicted > expected
  under_prediction predicted < expected
  injected_error   a deliberately injected wrong prediction (test hook)
"""
import math

TOL = 1e-9
REASONS = ("correct", "over_prediction", "under_prediction", "injected_error")


def fit_linear(train):
    if not train:
        raise ValueError("training records must be non-empty")
    xs = [r["x1"] for r in train]
    ys = [r["target"] for r in train]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        raise ValueError("x1 has no variance in the training data")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return {"slope": slope, "intercept": my - slope * mx}


def predict(model, record):
    return model["slope"] * record["x1"] + model["intercept"]


def classify(expected, predicted, tol=TOL):
    if abs(predicted - expected) <= tol:
        return "correct"
    return "over_prediction" if predicted > expected else "under_prediction"


def build_report(records, splits, model, injected=None):
    """injected: {record_id: wrong_predicted_value} to make the report testable."""
    injected = dict(injected or {})
    known = {r["record_id"] for r in records}
    for rid, value in injected.items():
        if rid not in known:
            raise ValueError(f"unknown injected record id: {rid}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"injected value for {rid} must be a finite number")
    split_of = {rid: s for s, ids in splits.items() for rid in ids}
    rows = []
    for r in records:
        rid = r["record_id"]
        expected = float(r["target"])
        predicted = float(injected[rid]) if rid in injected else float(predict(model, r))
        error = predicted - expected
        reason = classify(expected, predicted)
        if rid in injected and reason != "correct":
            reason = "injected_error"
        rows.append({
            "record_id": rid,
            "split": split_of[rid],
            "expected": expected,
            "predicted": predicted,
            "error": error,
            "abs_error": abs(error),
            "reason_category": reason,
        })
    err = [x for x in rows if x["reason_category"] != "correct"]
    summary = {
        "n_total": len(rows),
        "n_correct": len(rows) - len(err),
        "n_errors": len(err),
        "mae": sum(x["abs_error"] for x in rows) / len(rows),
        "by_split": {s: {"n": sum(1 for x in rows if x["split"] == s),
                         "n_errors": sum(1 for x in err if x["split"] == s)}
                     for s in splits},
        "by_reason": {c: sum(1 for x in rows if x["reason_category"] == c)
                      for c in REASONS},
        "error_record_ids": [x["record_id"] for x in err],
    }
    return {"rows": rows, "summary": summary}


def render_text(report):
    lines = [f"{'record_id':<10}{'split':<7}{'expected':>10}{'predicted':>11}{'error':>9}  reason"]
    for x in report["rows"]:
        lines.append(f"{x['record_id']:<10}{x['split']:<7}{x['expected']:>10.2f}"
                     f"{x['predicted']:>11.2f}{x['error']:>9.2f}  {x['reason_category']}")
    s = report["summary"]
    lines.append(f"total={s['n_total']} correct={s['n_correct']} errors={s['n_errors']} "
                 f"mae={s['mae']:.4f} error_ids={s['error_record_ids']}")
    return "\n".join(lines)
