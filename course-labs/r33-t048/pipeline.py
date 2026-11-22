"""
Course Lab R33-T048: Regression Error-Analysis Report
Generates structured residual analysis and fault categorization with injected anomaly detection.
Formula: target = 2 * x1 + 1.
"""
from typing import List, Dict, Any

class RegressionErrorAnalyzer:
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def predict_formula(self, x1: float) -> float:
        return 2.0 * float(x1) + 1.0

    def generate_report(self, records: List[Dict[str, Any]], injected_error_id: str = "r6", injected_value: float = 99.0) -> Dict[str, Any]:
        """
        Runs predictions. On injected_error_id, deliberately injects a wrong prediction
        so error categorization is deterministically testable.
        """
        rows = []
        counts = {"EXACT_MATCH": 0, "ACCEPTABLE_RESIDUAL": 0, "OUTLIER_ANOMALY": 0}

        for r in records:
            rid = r["record_id"]
            expected = float(r["target"])
            if rid == injected_error_id:
                pred = injected_value
                category = "OUTLIER_ANOMALY"
            else:
                pred = self.predict_formula(r["x1"])
                residual = abs(expected - pred)
                if residual == 0.0:
                    category = "EXACT_MATCH"
                elif residual <= self.tolerance:
                    category = "ACCEPTABLE_RESIDUAL"
                else:
                    category = "OUTLIER_ANOMALY"

            counts[category] += 1
            rows.append({
                "record_id": rid,
                "split": r.get("split"),
                "expected": expected,
                "predicted": pred,
                "residual": abs(expected - pred),
                "reason_category": category
            })

        return {
            "detailed_rows": rows,
            "summary_counts": counts,
            "total_records": len(records),
            "injected_error_id": injected_error_id
        }
