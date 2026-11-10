"""Bounded CPU smoke run: tiny logistic baseline on the synthetic fixture.
Asserts interface correctness (finite loss, output shape, step count),
NOT useful accuracy."""
import math
import random

MAX_SAMPLES = 32
MAX_STEPS = 20
SEED = 0


def _flat(pixels):
    return [p / 255.0 for row in pixels for p in row]


def _sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def scalar_loss(y, p, eps=1e-7):
    """Binary cross-entropy for one scalar label/probability."""
    p = min(max(p, eps), 1 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def predict_proba(w, b, records):
    return [_sigmoid(sum(wi * xi for wi, xi in zip(w, _flat(r["pixels"]))) + b)
            for r in records]


def train(records, steps=MAX_STEPS, lr=0.5, seed=SEED):
    if not records:
        raise ValueError("records must be non-empty")
    if len(records) > MAX_SAMPLES:
        raise ValueError(f"at most {MAX_SAMPLES} samples allowed")
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be in 1..{MAX_STEPS}")
    rng = random.Random(seed)
    w = [rng.uniform(-0.1, 0.1) for _ in range(16)]
    b = 0.0
    X = [_flat(r["pixels"]) for r in records]
    y = [r["label"] for r in records]
    losses = []
    for _ in range(steps):
        p = [_sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b) for x in X]
        losses.append(sum(scalar_loss(t, q) for t, q in zip(y, p)) / len(y))
        gw = [sum((q - t) * x[j] for q, t, x in zip(p, y, X)) / len(y) for j in range(16)]
        gb = sum(q - t for q, t in zip(p, y)) / len(y)
        w = [wi - lr * g for wi, g in zip(w, gw)]
        b -= lr * gb
    return {"w": w, "b": b, "losses": losses, "steps": len(losses)}
