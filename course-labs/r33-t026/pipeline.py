"""Bounded CPU smoke run pipeline for R33-T026.

Runs a tiny CPU-only classification model bounded to <= 32 samples,
fixed seed, and <= 20 update steps.
"""
import math
from typing import List, Dict, Any, Tuple

class BoundedCPUSmokeClassifier:
    """A minimal deterministic logistic regression smoke classifier."""
    def __init__(self, seed: int = 42, max_steps: int = 15, lr: float = 0.05) -> None:
        self.seed = seed
        self.max_steps = min(max_steps, 20)  # Bound strictly to at most 20 steps
        self.lr = lr
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0
        self.recorded_steps = 0
        self.final_loss = float("inf")
        self.is_constant_model_label = "deterministic_bounded_cpu_smoke_model"

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(min(z, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-z))

    def train_smoke_run(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs training bounded to <= 32 samples and recorded step count."""
        if len(records) > 32:
            raise ValueError("Smoke run must not exceed 32 samples")
        n = len(records)
        if n == 0:
            raise ValueError("Cannot train on empty records")

        losses = []
        for step in range(self.max_steps):
            total_loss = 0.0
            dw1, dw2, db = 0.0, 0.0, 0.0
            for r in records:
                x1, x2, y = float(r["x1"]), float(r["x2"]), float(r["label"])
                p = self._sigmoid(self.w1 * x1 + self.w2 * x2 + self.b)
                # Binary cross-entropy
                eps = 1e-12
                total_loss += -(y * math.log(p + eps) + (1.0 - y) * math.log(1.0 - p + eps))
                err = p - y
                dw1 += err * x1
                dw2 += err * x2
                db += err

            # Gradient update
            self.w1 -= self.lr * (dw1 / n)
            self.w2 -= self.lr * (dw2 / n)
            self.b -= self.lr * (db / n)
            self.recorded_steps += 1
            losses.append(total_loss / n)

        self.final_loss = losses[-1] if losses else 0.0
        return {
            "model_type": self.is_constant_model_label,
            "recorded_steps": self.recorded_steps,
            "final_loss": self.final_loss,
            "is_finite_loss": math.isfinite(self.final_loss),
            "sample_count": n
        }

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        preds = []
        for r in records:
            x1, x2 = float(r["x1"]), float(r["x2"])
            p = self._sigmoid(self.w1 * x1 + self.w2 * x2 + self.b)
            preds.append(1 if p >= 0.5 else 0)
        return preds
