"""
Course Lab R33-T022: Honest Majority-Class Baseline
Pipeline verification fixture for Performance Measures (03_classification.ipynb).
Synthetic CPU fixture only - not evidence of real-world or clinical performance.
"""

from typing import List, Dict, Any, Tuple
from collections import Counter

class MajorityClassBaseline:
    """
    Computes majority class strictly from training data and predicts on held-out test data.
    """
    def __init__(self):
        self.majority_class = None
        self.fitted = False

    def fit(self, train_records: List[Dict[str, Any]]) -> "MajorityClassBaseline":
        if not train_records:
            raise ValueError("Training records cannot be empty.")
        labels = [r["label"] for r in train_records]
        counts = Counter(labels)
        # In case of tie, pick smaller label deterministically
        max_freq = max(counts.values())
        candidates = [lbl for lbl, c in counts.items() if c == max_freq]
        self.majority_class = min(candidates)
        self.fitted = True
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")
        return [self.majority_class for _ in records]

    def evaluate(self, records: List[Dict[str, Any]], predictions: List[int]) -> Dict[str, float]:
        if len(records) != len(predictions):
            raise ValueError("Mismatch between records and predictions length.")
        if not records:
            raise ValueError("Records cannot be empty.")
        correct = sum(1 for r, p in zip(records, predictions) if r["label"] == p)
        accuracy = correct / len(records)
        return {"accuracy": accuracy, "count": len(records), "correct": correct}

def stratified_split(records: List[Dict[str, Any]], train_ratio: float = 0.5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records deterministically while preserving exact class distribution.
    """
    by_class = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)
    train, test = [], []
    for lbl in sorted(by_class.keys()):
        cls_records = sorted(by_class[lbl], key=lambda x: x["record_id"])
        n_train = int(len(cls_records) * train_ratio)
        train.extend(cls_records[:n_train])
        test.extend(cls_records[n_train:])
    train.sort(key=lambda x: x["record_id"])
    test.sort(key=lambda x: x["record_id"])
    return train, test
