"""Error analysis report generator for R33-T025.

Generates structured error report comparing expected vs predicted assignments,
with stable record IDs, reason categories, and aggregate error statistics.
"""
import math
from typing import List, Dict, Any, Tuple

class ErrorAnalysisPipeline:
    """Computes nearest-centroid assignments and generates error reports."""
    def __init__(self) -> None:
        self.centroids = [(0.0, 0.0), (5.0, 5.0)]

    @staticmethod
    def _dist(p: Tuple[float, float], c: Tuple[float, float]) -> float:
        return math.hypot(p[0] - c[0], p[1] - c[1])

    def predict_assignments(self, records: List[Dict[str, Any]]) -> List[int]:
        assignments = []
        for r in records:
            p = (float(r["x1"]), float(r["x2"]))
            d0 = self._dist(p, self.centroids[0])
            d1 = self._dist(p, self.centroids[1])
            assignments.append(0 if d0 <= d1 else 1)
        return assignments

    def generate_error_report(
        self,
        records: List[Dict[str, Any]],
        predictions: List[int]
    ) -> Dict[str, Any]:
        if len(records) != len(predictions):
            raise ValueError("Records and predictions count must match")

        error_rows = []
        for r, pred in zip(records, predictions):
            expected = r["expected_cluster"]
            if pred != expected:
                error_rows.append({
                    "record_id": r["record_id"],
                    "expected": expected,
                    "predicted": pred,
                    "reason_category": "centroid_boundary_misclassification"
                })

        return {
            "total_records": len(records),
            "total_errors": len(error_rows),
            "error_rate": len(error_rows) / len(records) if records else 0.0,
            "error_details": error_rows
        }
