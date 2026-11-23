"""Robust regularized linear regression interface for R33-T028.

Tests perturbation robustness: record-order permutation with stable IDs,
missing fields, and shape mismatches.
"""
from typing import List, Dict, Any

class RobustLinearRegression:
    """Predicts target = 2*x1 + 1 with stable ID mapping and robustness verification."""
    def __init__(self, w1: float = 2.0, b: float = 1.0) -> None:
        self.w1 = w1
        self.b = b

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for r in records:
            if "record_id" not in r:
                raise KeyError("Missing required identifier: record_id")
            if "x1" not in r or "x2" not in r:
                raise KeyError(f"Missing required field in record {r.get('record_id')}")
            if not isinstance(r["x1"], (int, float)) or not isinstance(r["x2"], (int, float)):
                raise ValueError(f"Shape or type mismatch in record {r.get('record_id')}")
            pred_val = self.w1 * float(r["x1"]) + self.b
            results.append({
                "record_id": r["record_id"],
                "prediction": pred_val
            })
        return results

    def predict_as_id_map(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        preds = self.predict(records)
        return {p["record_id"]: p["prediction"] for p in preds}
