"""
Course Lab R33-T049: Input Perturbation Robustness
Tests permutation invariance under record-order changes and enforces input validation fallbacks.
"""
from typing import List, Dict, Any

class RobustImagePipeline:
    def __init__(self):
        pass

    def predict_sample(self, record: Dict[str, Any]) -> int:
        if "record_id" not in record or not record["record_id"]:
            raise KeyError("Record must contain record_id")
        if "image" not in record:
            raise KeyError("Record must contain image")

        img = record["image"]
        if not isinstance(img, list) or len(img) != 4 or not all(isinstance(r, list) and len(r) == 4 for r in img):
            raise ValueError(f"Shape mismatch: expected 4x4 matrix, got {type(img)}")

        # Deterministic check: diagonal pattern vs all zero
        diagonal_sum = sum(img[i][i] for i in range(4))
        return 1 if diagonal_sum > 0 else 0

    def predict_dataset_indexed(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Returns predictions indexed by record_id. Invariant to input ordering.
        """
        results = {}
        for r in records:
            rid = r["record_id"]
            results[rid] = self.predict_sample(r)
        return results
