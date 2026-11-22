"""
Course Lab R33-T050: Bounded CPU Image Pipeline Smoke Test
Builds a tiny CPU-only linear classifier smoke run with bounded steps (<= 20) and finite loss.
"""
import math
from typing import List, Dict, Any, Tuple

class TinyImagePipeline:
    def __init__(self, seed: int = 42, lr: float = 0.01):
        self.seed = seed
        self.lr = lr
        self.weights = [0.0] * 16
        self.bias = 0.0
        self.step_count = 0
        self.loss_history: List[float] = []

    def _sigmoid(self, z: float) -> float:
        z_clamped = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def train_smoke_epoch(self, records: List[Dict[str, Any]], max_steps: int = 20) -> Dict[str, Any]:
        """
        Runs at most 1 epoch (at most max_steps steps) on small CPU data.
        Verifies finite loss and records exact step count.
        """
        if len(records) > 32:
            raise ValueError(f"Smoke test bounds exceeded: max 32 samples, got {len(records)}")

        epoch_loss = 0.0
        steps = 0

        for r in records:
            if steps >= max_steps:
                break
            x = [float(p) / 255.0 for row in r["image"] for p in row]
            y = float(r["label"])

            z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            y_hat = self._sigmoid(z)

            eps = 1e-12
            loss = -(y * math.log(max(eps, y_hat)) + (1.0 - y) * math.log(max(eps, 1.0 - y_hat)))
            epoch_loss += loss

            # Gradient step
            err = y_hat - y
            for i in range(16):
                self.weights[i] -= self.lr * err * x[i]
            self.bias -= self.lr * err

            steps += 1
            self.step_count += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        self.loss_history.append(avg_loss)

        return {
            "avg_loss": avg_loss,
            "steps_executed": steps,
            "total_steps": self.step_count,
            "is_finite_loss": math.isfinite(avg_loss)
        }

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        preds = []
        for r in records:
            x = [float(p) / 255.0 for row in r["image"] for p in row]
            z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            y_hat = self._sigmoid(z)
            preds.append(1 if y_hat >= 0.5 else 0)
        return preds
