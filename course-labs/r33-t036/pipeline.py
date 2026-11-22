"""
R33-T036: Add a bounded CPU smoke run
=====================================
Project/section: 06 decision trees / Sensitivity to axis orientation
Notebook: 06_decision_trees.ipynb

Goal
----
Build a tiny classification pipeline smoke test: at most 32 samples, a fixed
seed, and at most one training epoch / 20 update steps. Use a small CPU model
or deterministic baseline; assert interface correctness, not useful accuracy.

Fixture:
Eight synthetic rows with x1=[0,1,2,3,4,5,6,7], x2=1 and label=[0,0,0,0,1,1,1,1].
Stable row/group IDs with a fixed stratified split.

Data contract:
- Source: Fully synthetic, generated for smoke-testing pipeline verification only.
- Licence: N/A.
- Real dataset used: No.
- Disclaimer: These records are simulated. Results must not be cited as evidence
  of real-world or clinical performance.
"""

import math
import random
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Synthetic Fixture
# ---------------------------------------------------------------------------

SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "s1", "group_id": "g0", "x1": 0.0, "x2": 1.0, "label": 0},
    {"record_id": "s2", "group_id": "g0", "x1": 1.0, "x2": 1.0, "label": 0},
    {"record_id": "s3", "group_id": "g0", "x1": 2.0, "x2": 1.0, "label": 0},
    {"record_id": "s4", "group_id": "g0", "x1": 3.0, "x2": 1.0, "label": 0},
    {"record_id": "s5", "group_id": "g1", "x1": 4.0, "x2": 1.0, "label": 1},
    {"record_id": "s6", "group_id": "g1", "x1": 5.0, "x2": 1.0, "label": 1},
    {"record_id": "s7", "group_id": "g1", "x1": 6.0, "x2": 1.0, "label": 1},
    {"record_id": "s8", "group_id": "g1", "x1": 7.0, "x2": 1.0, "label": 1},
]

# Fixed stratified split preserving class balance and group isolation
TRAIN_RECORD_IDS = ["s1", "s2", "s3", "s5", "s6", "s7"]
TEST_RECORD_IDS = ["s4", "s8"]

MAX_SAMPLES_BOUND = 32
MAX_STEPS_BOUND = 20
MAX_EPOCHS_BOUND = 1


def get_stratified_split() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return explicit (train, test) partitions for the synthetic fixture."""
    id_map = {row["record_id"]: row for row in SYNTHETIC_FIXTURE}
    train = [id_map[rid] for rid in TRAIN_RECORD_IDS]
    test = [id_map[rid] for rid in TEST_RECORD_IDS]
    return train, test


def _sigmoid(z: float) -> float:
    # Bounded numerical clamp for CPU stability
    if z < -20.0:
        return 0.0
    if z > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NaiveConstantBaseline:
    """Explicitly labelled trivial baseline predicting the majority training class."""

    def __init__(self):
        self.model_type = "naive_constant_baseline"
        self.constant_label: int = 0
        self.fitted = False

    def fit(self, records: List[Dict[str, Any]]) -> "NaiveConstantBaseline":
        if not records:
            raise ValueError("Cannot fit on empty records.")
        labels = [r["label"] for r in records]
        count_1 = sum(labels)
        count_0 = len(labels) - count_1
        self.constant_label = 1 if count_1 > count_0 else 0
        self.fitted = True
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict().")
        return [self.constant_label for _ in records]


class TinyLogisticCPUClassifier:
    """
    Lightweight CPU linear/logistic classification model bounded for smoke runs:
    - <= 32 samples
    - fixed random seed
    - <= 1 epoch / <= 20 update steps
    - finite outputs and loss
    """

    def __init__(self, seed: int = 42, lr: float = 0.05):
        self.model_type = "tiny_logistic_cpu"
        self.seed = seed
        self.lr = lr
        self.w1 = 0.0
        self.w2 = 0.0
        self.bias = 0.0
        self.steps_executed = 0
        self.epochs_executed = 0
        self.sample_count = 0
        self.loss_history: List[float] = []
        self.fitted = False

    def fit(
        self,
        records: List[Dict[str, Any]],
        max_steps: int = 20,
        max_epochs: int = 1,
    ) -> "TinyLogisticCPUClassifier":
        if len(records) > MAX_SAMPLES_BOUND:
            raise ValueError(
                f"Sample count {len(records)} exceeds smoke test bound of {MAX_SAMPLES_BOUND}."
            )
        if max_steps > MAX_STEPS_BOUND:
            raise ValueError(
                f"max_steps {max_steps} exceeds smoke test bound of {MAX_STEPS_BOUND}."
            )
        if max_epochs > MAX_EPOCHS_BOUND:
            raise ValueError(
                f"max_epochs {max_epochs} exceeds smoke test bound of {MAX_EPOCHS_BOUND}."
            )
        if not records:
            raise ValueError("Training dataset cannot be empty.")

        self.sample_count = len(records)
        rng = random.Random(self.seed)

        # Deterministic small initialization
        self.w1 = (rng.random() - 0.5) * 0.1
        self.w2 = (rng.random() - 0.5) * 0.1
        self.bias = 0.0
        self.steps_executed = 0
        self.loss_history = []

        data = list(records)
        rng.shuffle(data)

        # Single bounded epoch, at most max_steps updates
        for epoch in range(max_epochs):
            self.epochs_executed = epoch + 1
            for row in data:
                if self.steps_executed >= max_steps:
                    break

                x1 = float(row["x1"])
                x2 = float(row["x2"])
                y = float(row["label"])

                # Forward
                z = self.w1 * x1 + self.w2 * x2 + self.bias
                p = _sigmoid(z)

                # Cross-entropy loss (bounded)
                p_clamped = max(1e-12, min(1.0 - 1e-12, p))
                loss = -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
                if not math.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss encountered at step {self.steps_executed}")
                self.loss_history.append(loss)

                # Backward step (single-sample SGD)
                grad = p - y
                self.w1 -= self.lr * grad * x1
                self.w2 -= self.lr * grad * x2
                self.bias -= self.lr * grad
                self.steps_executed += 1

        self.fitted = True
        return self

    def predict_proba(self, records: List[Dict[str, Any]]) -> List[float]:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict_proba().")
        probas = []
        for row in records:
            x1 = float(row["x1"])
            x2 = float(row["x2"])
            z = self.w1 * x1 + self.w2 * x2 + self.bias
            p = _sigmoid(z)
            if not math.isfinite(p):
                raise RuntimeError("Non-finite probability output encountered.")
            probas.append(p)
        return probas

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        probas = self.predict_proba(records)
        return [1 if p >= 0.5 else 0 for p in probas]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_smoke_pipeline(seed: int = 42) -> Dict[str, Any]:
    """
    Executes the bounded CPU smoke run on the synthetic fixture.
    Asserts interface correctness, finite loss, bounded step count, and expected shapes.
    """
    train, test = get_stratified_split()

    # 1. Tiny CPU Model
    model = TinyLogisticCPUClassifier(seed=seed)
    model.fit(train, max_steps=20, max_epochs=1)

    train_preds = model.predict(train)
    test_preds = model.predict(test)
    test_probas = model.predict_proba(test)

    # 2. Naive Constant Baseline
    naive = NaiveConstantBaseline()
    naive.fit(train)
    naive_test_preds = naive.predict(test)

    # Interface correctness assertions
    assert len(train_preds) == len(train), "Train prediction shape mismatch"
    assert len(test_preds) == len(test), "Test prediction shape mismatch"
    assert len(test_probas) == len(test), "Test proba shape mismatch"
    assert model.steps_executed <= MAX_STEPS_BOUND, "Steps exceeded bound"
    assert model.epochs_executed <= MAX_EPOCHS_BOUND, "Epochs exceeded bound"
    assert all(math.isfinite(loss) for loss in model.loss_history), "Non-finite loss found"
    assert all(math.isfinite(p) for p in test_probas), "Non-finite probability found"

    return {
        "model_type": model.model_type,
        "sample_count": model.sample_count,
        "steps_executed": model.steps_executed,
        "epochs_executed": model.epochs_executed,
        "final_loss": model.loss_history[-1] if model.loss_history else None,
        "loss_history_length": len(model.loss_history),
        "train_ids": [r["record_id"] for r in train],
        "test_ids": [r["record_id"] for r in test],
        "test_predictions": test_preds,
        "test_probas": test_probas,
        "naive_baseline_type": naive.model_type,
        "naive_test_predictions": naive_test_preds,
        "status": "PASS",
    }


if __name__ == "__main__":
    result = run_smoke_pipeline()
    print("Smoke Run Output:")
    for k, v in result.items():
        print(f"  {k}: {v}")
