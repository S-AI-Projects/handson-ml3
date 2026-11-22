"""
R33-T038: Explain an error case
===============================
Project/section: 07 ensemble learning and random forests / AdaBoost
Notebook: 07_ensemble_learning_and_random_forests.ipynb

Goal
----
Create a small error-analysis report for this classification fixture:
stable record ID, expected value, predicted value and a reason category.
Add one deliberately wrong prediction so the report is testable.

Fixture:
Eight synthetic rows with x1=[0,1,2,3,4,5,6,7], x2=1 and label=[0,0,0,0,1,1,1,1].
Stable row/group IDs with a fixed stratified split.

Data contract:
- Source: Fully synthetic, generated for error-analysis pipeline verification only.
- Licence: N/A.
- Real dataset used: No.
- Disclaimer: These records are simulated. Results must not be cited as evidence
  of real-world or clinical performance.
"""

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Synthetic Fixture
# ---------------------------------------------------------------------------

SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "s1", "group_id": "g0", "x1": 0.0, "x2": 1.0, "label": 0},
    {"record_id": "s2", "group_id": "g0", "x1": 1.0, "x2": 1.0, "label": 0},
    {"record_id": "s3", "group_id": "g0", "x1": 2.0, "x2": 1.0, "label": 0},
    {"record_id": "s4", "group_id": "g0", "x1": 3.0, "x2": 1.0, "label": 0},
    {"record_id": "s5", "group_id": "g1", "x1": 4.0, "x2": 1.0, "label": 1},
    {"record_id": "s6", "group_id": "g1", "x1": 5.0, "x2": 1.0, "label": 1},
    {"record_id": "s7", "group_id": "g1", "x1": 6.0, "x2": 1.0, "label": 1},
    {"record_id": "s8", "group_id": "g1", "x1": 7.0, "x2": 1.0, "label": 1},
]

TRAIN_RECORD_IDS = ["s1", "s2", "s3", "s5", "s6", "s7"]
TEST_RECORD_IDS = ["s4", "s8"]

INJECTED_ERROR_TARGET_ID = "s4"
INJECTED_ERROR_WRONG_PRED = 1
INJECTED_ERROR_REASON = "decision_boundary_proximity"


# ---------------------------------------------------------------------------
# Baseline Decision Stump Classifier (AdaBoost weak learner)
# ---------------------------------------------------------------------------

class DecisionStumpClassifier:
    """Decision stump splitting on x1 feature separating class 0 and 1."""

    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
        self.fitted = False

    def fit(self, records: List[Dict[str, Any]]) -> "DecisionStumpClassifier":
        if not records:
            raise ValueError("Cannot fit on empty records.")
        x1_0 = [r["x1"] for r in records if r["label"] == 0]
        x1_1 = [r["x1"] for r in records if r["label"] == 1]
        if x1_0 and x1_1:
            self.threshold = (max(x1_0) + min(x1_1)) / 2.0
        self.fitted = True
        return self

    def predict_record(self, record: Dict[str, Any]) -> int:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return 1 if float(record["x1"]) > self.threshold else 0

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        return {r["record_id"]: self.predict_record(r) for r in records}


# ---------------------------------------------------------------------------
# Error Analysis Engine
# ---------------------------------------------------------------------------

def categorize_error_reason(
    record: Dict[str, Any],
    expected: int,
    predicted: int,
    threshold: float = 3.0,
) -> str:
    """Categorizes the reason for classification error."""
    if expected == predicted:
        return "none"

    distance_to_boundary = abs(float(record["x1"]) - threshold)
    if distance_to_boundary <= 1.0:
        return "decision_boundary_proximity"
    elif float(record["x1"]) in (0.0, 7.0):
        return "extreme_tail_outlier"
    else:
        return "feature_ambiguity"


def inject_prediction_error(
    predictions: Dict[str, int],
    target_id: str = INJECTED_ERROR_TARGET_ID,
    wrong_value: int = INJECTED_ERROR_WRONG_PRED,
) -> Dict[str, int]:
    """Injects one deliberately wrong prediction to make error report testable."""
    if target_id not in predictions:
        raise KeyError(f"Target record_id '{target_id}' not found in predictions.")
    tampered = dict(predictions)
    tampered[target_id] = wrong_value
    return tampered


def generate_error_report(
    records: List[Dict[str, Any]],
    predictions: Dict[str, int],
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Constructs a structured error-analysis report.

    Requirements:
    - Correct predictions MUST NOT appear in error_rows.
    - The injected wrong prediction MUST appear in error_rows with the correct record ID.
    - Aggregate error count MUST match the length of detailed error_rows.
    """
    if len(records) != len(predictions):
        raise ValueError("Record count must match predictions dictionary size.")

    error_rows = []
    correct_count = 0

    for r in records:
        rid = r["record_id"]
        if rid not in predictions:
            raise KeyError(f"Missing prediction for record_id '{rid}'.")

        expected = int(r["label"])
        predicted = int(predictions[rid])

        if expected != predicted:
            reason = categorize_error_reason(r, expected, predicted, threshold)
            error_rows.append(
                {
                    "record_id": rid,
                    "group_id": r.get("group_id", "unknown"),
                    "x1": float(r["x1"]),
                    "x2": float(r["x2"]),
                    "expected_value": expected,
                    "predicted_value": predicted,
                    "reason_category": reason,
                    "margin_to_boundary": round(abs(float(r["x1"]) - threshold), 4),
                }
            )
        else:
            correct_count += 1

    total_records = len(records)
    total_errors = len(error_rows)
    accuracy = correct_count / total_records if total_records > 0 else 0.0

    return {
        "total_records": total_records,
        "total_correct": correct_count,
        "total_errors": total_errors,
        "accuracy": round(accuracy, 4),
        "error_rate": round(total_errors / total_records, 4) if total_records > 0 else 0.0,
        "error_rows": error_rows,
    }


def run_error_analysis_pipeline() -> Dict[str, Any]:
    """End-to-end execution of model fitting, error injection, and error reporting."""
    train_records = [r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS]

    # Fit weak learner
    stump = DecisionStumpClassifier()
    stump.fit(train_records)

    # Clean predictions (all 8 correct)
    clean_preds = stump.predict(SYNTHETIC_FIXTURE)

    # Injected error prediction on target record s4
    tampered_preds = inject_prediction_error(
        clean_preds,
        target_id=INJECTED_ERROR_TARGET_ID,
        wrong_value=INJECTED_ERROR_WRONG_PRED,
    )

    # Generate reports
    clean_report = generate_error_report(SYNTHETIC_FIXTURE, clean_preds, stump.threshold)
    error_report = generate_error_report(SYNTHETIC_FIXTURE, tampered_preds, stump.threshold)

    return {
        "threshold": stump.threshold,
        "clean_report": clean_report,
        "error_report": error_report,
        "injected_target": INJECTED_ERROR_TARGET_ID,
        "injected_reason": INJECTED_ERROR_REASON,
    }


if __name__ == "__main__":
    res = run_error_analysis_pipeline()
    print("Error Analysis Pipeline Output:")
    print(f"Clean Errors: {res['clean_report']['total_errors']}")
    print(f"Injected Errors: {res['error_report']['total_errors']}")
    print("Error Rows:")
    for row in res["error_report"]["error_rows"]:
        print(f"  {row}")
