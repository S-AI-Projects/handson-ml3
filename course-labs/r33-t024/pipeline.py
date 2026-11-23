"""Evaluation metric implementation and cross-checking for R33-T024.

Explicitly defines precision, recall, and zero-denominator handling.
"""
from typing import List, Dict, Any, Tuple

def compute_precision_recall(
    y_true: List[int],
    y_pred: List[int],
    zero_division: float = 0.0
) -> Tuple[float, float]:
    """Calculates precision and recall with defined zero-denominator behavior.
    
    tp: true positives, fp: false positives, fn: false negatives
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Truth and prediction lengths must match")
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    return precision, recall

def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    """Computes Mean Absolute Error for regression."""
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths must match")
    if not y_true:
        raise ValueError("Cannot compute MAE on empty lists")
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

class ImageClassifierMetricEvaluator:
    """Evaluates classifier predictions on image fixture."""
    def __init__(self, zero_division: float = 0.0) -> None:
        self.zero_division = zero_division

    def evaluate(self, records: List[Dict[str, Any]], predictions: List[int]) -> Dict[str, float]:
        y_true = [r["label"] for r in records]
        prec, rec = compute_precision_recall(y_true, predictions, zero_division=self.zero_division)
        return {
            "precision": prec,
            "recall": rec,
            "sample_count": float(len(records))
        }
