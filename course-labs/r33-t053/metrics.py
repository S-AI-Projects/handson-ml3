"""Evaluation metrics with explicitly defined behaviour.

Zero-denominator rule: if a ratio's denominator is 0 it returns 0.0
(zero_division parameter, default 0.0).
Averaging: 'binary' (positive class only), 'macro' (unweighted mean of
per-class scores), 'micro' (pooled TP / pooled denominators).
"""


def _div(num, den, zero_division=0.0):
    return zero_division if den == 0 else num / den


def _check(truth, pred):
    if len(truth) != len(pred):
        raise ValueError("truth and prediction must have the same length")
    if len(truth) == 0:
        raise ValueError("inputs must be non-empty")


def _counts(truth, pred, cls):
    tp = sum(1 for t, p in zip(truth, pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(truth, pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(truth, pred) if t == cls and p != cls)
    return tp, fp, fn


def _score(kind, truth, pred, average, positive, zero_division):
    _check(truth, pred)

    def one(c):
        tp, fp, fn = _counts(truth, pred, c)
        other = fp if kind == "precision" else fn
        return tp, tp + other

    if average == "binary":
        tp, den = one(positive)
        return _div(tp, den, zero_division)
    classes = sorted(set(truth) | set(pred))
    if average == "macro":
        return sum(_div(*one(c), zero_division) for c in classes) / len(classes)
    if average == "micro":
        tps, dens = zip(*(one(c) for c in classes))
        return _div(sum(tps), sum(dens), zero_division)
    raise ValueError(f"unknown average: {average}")


def precision(truth, pred, average="binary", positive=1, zero_division=0.0):
    return _score("precision", truth, pred, average, positive, zero_division)


def recall(truth, pred, average="binary", positive=1, zero_division=0.0):
    return _score("recall", truth, pred, average, positive, zero_division)


def mae(truth, pred):
    _check(truth, pred)
    return sum(abs(t - p) for t, p in zip(truth, pred)) / len(truth)


def leaky_relu(x, alpha=0.2):
    return x if x > 0 else alpha * x


def predict(pixels):
    """Tiny deterministic scorer using Leaky ReLU: diagonal mean - off-diagonal mean."""
    diag = [pixels[i][i] for i in range(4)]
    off = [pixels[i][j] for i in range(4) for j in range(4) if i != j]
    score = leaky_relu(sum(diag) / 4 - sum(off) / 12)
    return 1 if score > 0 else 0
