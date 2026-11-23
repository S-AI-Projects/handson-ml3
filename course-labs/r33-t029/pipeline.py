"""Safe JSON-based saved pipeline interface for R33-T029.

Provides serialization and deserialization using pure JSON (explicitly forbidding pickle).
Exposes prediction function with strict input schema validation.
"""
import json
import os
from typing import List, Dict, Any

REQUIRED_SCHEMA = ["record_id", "group_id", "x1", "x2"]

class SafeSavedClassificationPipeline:
    """Classification pipeline with safe JSON serialization and schema validation."""
    def __init__(self, weights: List[float] = None, bias: float = 0.0, threshold: float = 0.5) -> None:
        self.weights = weights or [0.5, 0.0]
        self.bias = bias
        self.threshold = threshold

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """Validates input schema and returns predictions."""
        preds = []
        for r in records:
            # Schema validation
            for field in REQUIRED_SCHEMA:
                if field not in r:
                    raise ValueError(f"Schema violation: missing required field '{field}' in record {r}")
            if not isinstance(r["x1"], (int, float)) or not isinstance(r["x2"], (int, float)):
                raise ValueError(f"Schema violation: non-numeric feature types in record {r}")

            score = self.weights[0] * float(r["x1"]) + self.weights[1] * float(r["x2"]) + self.bias
            preds.append(1 if score >= self.threshold else 0)
        return preds

    def save_json(self, filepath: str) -> None:
        """Serializes pipeline state to safe JSON."""
        state = {
            "model_format": "json_safe_pipeline_v1",
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "SafeSavedClassificationPipeline":
        """Loads pipeline from JSON, refusing untrusted formats like pickle."""
        if not filepath.endswith(".json"):
            raise ValueError("Untrusted format: only safe JSON serialization is permitted (no pickle)")
        with open(filepath, "r") as f:
            state = json.load(f)
        if state.get("model_format") != "json_safe_pipeline_v1":
            raise ValueError(f"Invalid model format header: {state.get('model_format')}")
        return cls(
            weights=state["weights"],
            bias=state["bias"],
            threshold=state["threshold"]
        )
