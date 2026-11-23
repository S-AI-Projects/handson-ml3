"""Preprocessing contract and nearest-centroid assignment for R33-T023.

Pipeline verification fixture: six 2D points with fixed centroids (0,0) and (5,5).
"""
import math
from typing import List, Dict, Tuple, Any

class RetainedScaler:
    """Scales numeric features using parameters fit strictly on training records."""
    def __init__(self) -> None:
        self.mean: List[float] = []
        self.std: List[float] = []
        self._fit_done: bool = False
        self._train_ids: set = set()

    def fit(self, records: List[Dict[str, Any]]) -> "RetainedScaler":
        if not records:
            raise ValueError("Cannot fit on empty dataset")
        for r in records:
            if not isinstance(r.get("x1"), (int, float)) or not isinstance(r.get("x2"), (int, float)):
                raise ValueError(f"Contamination or invalid schema in record: {r}")
        n = len(records)
        xs = [float(r["x1"]) for r in records]
        ys = [float(r["x2"]) for r in records]
        self.mean = [sum(xs) / n, sum(ys) / n]
        self.std = [
            math.sqrt(sum((x - self.mean[0]) ** 2 for x in xs) / n) or 1.0,
            math.sqrt(sum((y - self.mean[1]) ** 2 for y in ys) / n) or 1.0
        ]
        self._train_ids = {r["record_id"] for r in records}
        self._fit_done = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        if not self._fit_done:
            raise RuntimeError("Scaler must be fitted before calling transform")
        transformed = []
        for r in records:
            if not isinstance(r.get("x1"), (int, float)) or not isinstance(r.get("x2"), (int, float)):
                raise ValueError(f"Contamination or invalid schema: {r}")
            sx = (float(r["x1"]) - self.mean[0]) / self.std[0]
            sy = (float(r["x2"]) - self.mean[1]) / self.std[1]
            transformed.append((sx, sy))
        return transformed

class NearestCentroidClassifier:
    """Assigns points to nearest of fixed centroids (0,0) and (5,5)."""
    def __init__(self) -> None:
        self.centroids = [(0.0, 0.0), (5.0, 5.0)]

    @staticmethod
    def _dist(p: Tuple[float, float], c: Tuple[float, float]) -> float:
        return math.hypot(p[0] - c[0], p[1] - c[1])

    def assign(self, records: List[Dict[str, Any]]) -> List[int]:
        assignments = []
        for r in records:
            pt = (float(r["x1"]), float(r["x2"]))
            d0 = self._dist(pt, self.centroids[0])
            d1 = self._dist(pt, self.centroids[1])
            assignments.append(0 if d0 <= d1 else 1)
        return assignments

def split_manifest(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    # Stable deterministic split: r1, r2, r4, r5 for train; r3, r6 for test
    train_ids = {"r1", "r2", "r4", "r5"}
    train = [r for r in records if r["record_id"] in train_ids]
    test = [r for r in records if r["record_id"] not in train_ids]
    return {"train": train, "test": test}
