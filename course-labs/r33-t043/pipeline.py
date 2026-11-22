"""
Course Lab R33-T043: Minimal Majority-Class Baseline
Implements the simplest honest classification baseline without training learned weights.
"""
from typing import List, Dict, Any, Optional
from collections import Counter

class MajorityClassBaseline:
    def __init__(self):
        self.majority_class: Optional[int] = None
        self.class_counts: Dict[int, int] = {}

    def fit(self, records: List[Dict[str, Any]]) -> "MajorityClassBaseline":
        if not records:
            raise ValueError("Cannot fit on empty record collection")
        labels = [r["label"] for r in records]
        counts = Counter(labels)
        self.class_counts = dict(counts)
        # Deterministic tie-breaker: sort by count desc, then label asc
        sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self.majority_class = sorted_counts[0][0]
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        if self.majority_class is None:
            raise ValueError("Baseline has not been fitted")
        return [self.majority_class for _ in records]

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        if not records:
            raise ValueError("Cannot evaluate empty records")
        preds = self.predict(records)
        correct = sum(1 for p, r in zip(preds, records) if p == r["label"])
        acc = correct / len(records)
        return {
            "accuracy": acc,
            "total_samples": len(records),
            "majority_class": float(self.majority_class)
        }
