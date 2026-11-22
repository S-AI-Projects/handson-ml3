"""
Course Lab R33-T047: Safe Pickle-Free Model Persistence
Implements JSON-based model parameter storage, reload verification, and inference schema validation.
"""
import json
import os
from typing import List, Dict, Any

class SafePretrainedPipeline:
    def __init__(self, weights: List[float] = None, bias: float = 0.0, threshold: float = 0.5):
        # 16 weights for 4x4 image
        self.schema_version = "1.0.0"
        self.weights = weights if weights is not None else [1.0 if (i % 5 == 0) else -0.5 for i in range(16)]
        self.bias = bias
        self.threshold = threshold

    def predict_sample(self, image: List[List[float]]) -> int:
        if not isinstance(image, list) or len(image) != 4 or not all(isinstance(r, list) and len(r) == 4 for r in image):
            raise ValueError("Input sample must be a 4x4 2D numerical list")
        flat = [float(p) for row in image for p in row]
        score = sum(w * x for w, x in zip(self.weights, flat)) + self.bias
        return 1 if score >= self.threshold else 0

    def predict_dataset(self, records: List[Dict[str, Any]]) -> List[int]:
        return [self.predict_sample(r["image"]) for r in records]

    def save_json(self, filepath: str):
        if filepath.endswith(".pkl") or filepath.endswith(".pickle"):
            raise ValueError("Pickle persistence is prohibited for security reasons; use JSON format")
        payload = {
            "schema_version": self.schema_version,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "SafePretrainedPipeline":
        if filepath.endswith(".pkl") or filepath.endswith(".pickle"):
            raise ValueError("Untrusted pickle file loading is prohibited; use JSON format")
        with open(filepath, "r") as f:
            data = json.load(f)
        if data.get("schema_version") != "1.0.0":
            raise ValueError(f"Incompatible schema version: {data.get('schema_version')}")
        return cls(weights=data["weights"], bias=data["bias"], threshold=data["threshold"])
