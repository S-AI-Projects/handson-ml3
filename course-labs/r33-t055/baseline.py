"""Majority-class baseline for the classification contract.

Parameters are learned from TRAINING labels only.
Tie rule: if classes are tied, the smallest label wins (deterministic).
Metric: accuracy = correct / total.
"""
from collections import Counter


def fit_majority(train_labels):
    if not train_labels:
        raise ValueError("training labels must be non-empty")
    counts = Counter(train_labels)
    top = max(counts.values())
    return min(label for label, c in counts.items() if c == top)


def predict_majority(majority, n):
    if n < 0:
        raise ValueError("n must be non-negative")
    return [majority] * n


def accuracy(truth, pred):
    if len(truth) != len(pred):
        raise ValueError("truth and prediction must have the same length")
    if len(truth) == 0:
        raise ValueError("inputs must be non-empty")
    return sum(1 for t, p in zip(truth, pred) if t == p) / len(truth)
