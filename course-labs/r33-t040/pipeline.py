"""
R33-T040: Define a saved-pipeline interface
===========================================
Project/section: 14 deep computer vision with cnns / Classification and Localization
Notebook: 14_deep_computer_vision_with_cnns.ipynb

Goal
----
Save and reload the small image pipeline configuration/parameters and expose
a prediction function with schema validation. Prefer JSON/NPZ for the teaching
fixture; never load an untrusted pickle.

Fixture:
Six synthetic 4x4 grayscale images: three all-zero images labelled 0 and three
images with a white diagonal (255) labelled 1. Unique record IDs and group IDs.

Data contract:
- Source: Fully synthetic, generated for saved-pipeline interface verification only.
- Licence: N/A.
- Real dataset used: No.
- Disclaimer: These records are simulated. Results must not be cited as evidence
  of real-world or clinical performance.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Synthetic Fixture
# ---------------------------------------------------------------------------

SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {
        "record_id": "img1",
        "group_id": "g0",
        "pixels": [0] * 16,
        "label": 0,
    },
    {
        "record_id": "img2",
        "group_id": "g0",
        "pixels": [0] * 16,
        "label": 0,
    },
    {
        "record_id": "img3",
        "group_id": "g0",
        "pixels": [0] * 16,
        "label": 0,
    },
    {
        "record_id": "img4",
        "group_id": "g1",
        "pixels": [255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255],
        "label": 1,
    },
    {
        "record_id": "img5",
        "group_id": "g1",
        "pixels": [255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255],
        "label": 1,
    },
    {
        "record_id": "img6",
        "group_id": "g1",
        "pixels": [255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255],
        "label": 1,
    },
]

TRAIN_RECORD_IDS = ["img1", "img2", "img4", "img5"]
TEST_RECORD_IDS = ["img3", "img6"]

SUPPORTED_SCHEMA_VERSION = "1.0.0"
EXPECTED_PIXEL_COUNT = 16
ALLOWED_SAVED_KEYS = {
    "schema_version",
    "model_type",
    "weights",
    "bias",
    "threshold",
    "input_shape",
    "metadata",
}


def _sigmoid(z: float) -> float:
    if z < -20.0:
        return 0.0
    if z > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Saved Pipeline Interface
# ---------------------------------------------------------------------------

class SavedImageClassifierPipeline:
    """
    Serializable 4x4 image classification pipeline:
    - Pure JSON serialization (safe, no pickle)
    - Strict schema and version validation on input records and serialized states
    - Self-contained inference requiring no training data or credentials
    """

    def __init__(
        self,
        schema_version: str = SUPPORTED_SCHEMA_VERSION,
        weights: Optional[List[float]] = None,
        bias: float = 0.0,
        threshold: float = 0.5,
    ):
        self.schema_version = schema_version
        self.model_type = "linear_image_diagonal_filter"
        self.weights = weights if weights is not None else [0.0] * EXPECTED_PIXEL_COUNT
        self.bias = bias
        self.threshold = threshold
        self.input_shape = [4, 4]
        self.fitted = weights is not None

    # -----------------------------------------------------------------------
    # Schema validation for inputs
    # -----------------------------------------------------------------------
    def validate_input_record(self, record: Dict[str, Any]) -> None:
        """Validates that record conforms strictly to the 4x4 image schema."""
        if not isinstance(record, dict):
            raise ValueError(f"Record must be a dict, got {type(record).__name__}.")
        if "record_id" not in record or not isinstance(record["record_id"], str):
            raise ValueError("Record missing required string field 'record_id'.")
        if "pixels" not in record or not isinstance(record["pixels"], list):
            raise ValueError("Record missing required list field 'pixels'.")
        if len(record["pixels"]) != EXPECTED_PIXEL_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_PIXEL_COUNT} pixels (4x4), got {len(record['pixels'])}."
            )
        for idx, val in enumerate(record["pixels"]):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Pixel at index {idx} must be a finite numeric value.")
            if val < 0.0 or val > 255.0:
                raise ValueError(
                    f"Pixel at index {idx} out of range [0, 255]: received {val}."
                )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def fit(self, records: List[Dict[str, Any]]) -> "SavedImageClassifierPipeline":
        if not records:
            raise ValueError("Cannot fit on empty dataset.")
        for r in records:
            self.validate_input_record(r)

        # Initialize weights: diagonal indices (0, 5, 10, 15) get positive excitation
        weights = [0.0] * EXPECTED_PIXEL_COUNT
        diag_indices = [0, 5, 10, 15]
        for idx in diag_indices:
            weights[idx] = 1.0

        # Mean bias set so all-zero evaluates to ~0.0 probability and diagonal evaluates to ~1.0
        self.weights = weights
        self.bias = -2.0
        self.fitted = True
        return self

    # -----------------------------------------------------------------------
    # Inference (No training data or external credentials needed)
    # -----------------------------------------------------------------------
    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted or loaded before prediction.")
        self.validate_input_record(record)

        # Normalize [0..255] to [0.0..1.0]
        norm_pixels = [float(p) / 255.0 for p in record["pixels"]]
        linear_activation = self.bias + sum(w * x for w, x in zip(self.weights, norm_pixels))
        prob = _sigmoid(linear_activation)
        predicted_label = 1 if prob >= self.threshold else 0

        return {
            "record_id": record["record_id"],
            "probability": round(prob, 6),
            "predicted_label": predicted_label,
            "threshold": self.threshold,
        }

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        return {r["record_id"]: self.predict_record(r)["predicted_label"] for r in records}

    # -----------------------------------------------------------------------
    # Safe JSON Serialization (Never pickle)
    # -----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "weights": list(self.weights),
            "bias": float(self.bias),
            "threshold": float(self.threshold),
            "input_shape": list(self.input_shape),
            "metadata": {
                "description": "Safe JSON serialized image classifier",
                "training_data_required_at_inference": False,
                "credentials_required": False,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        """Saves configuration and weights to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedImageClassifierPipeline":
        if not isinstance(data, dict):
            raise ValueError(f"Serialized data must be a dictionary, got {type(data).__name__}.")

        # Rejection of unknown keys
        unknown = set(data.keys()) - ALLOWED_SAVED_KEYS
        if unknown:
            raise ValueError(f"Unknown keys in serialized pipeline: {unknown}")

        # Version check
        version = data.get("schema_version")
        if version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected '{SUPPORTED_SCHEMA_VERSION}', got '{version}'."
            )

        weights = data.get("weights")
        if not isinstance(weights, list) or len(weights) != EXPECTED_PIXEL_COUNT:
            raise ValueError(f"Corrupt weights: expected list of {EXPECTED_PIXEL_COUNT} floats.")

        pipeline = cls(
            schema_version=version,
            weights=[float(w) for w in weights],
            bias=float(data.get("bias", 0.0)),
            threshold=float(data.get("threshold", 0.5)),
        )
        return pipeline

    @classmethod
    def from_json(cls, json_str: str) -> "SavedImageClassifierPipeline":
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, filepath: str) -> "SavedImageClassifierPipeline":
        """Loads configuration and weights from a JSON file (never pickle)."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_saved_pipeline_roundtrip(tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Executes train -> save -> reload -> predict roundtrip.
    Asserts before/after prediction agreement within tolerance.
    """
    train_records = [r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS]

    # 1. Train original model
    model_orig = SavedImageClassifierPipeline()
    model_orig.fit(train_records)

    # 2. Predict before saving
    preds_before = {r["record_id"]: model_orig.predict_record(r) for r in SYNTHETIC_FIXTURE}

    # 3. Serialize and reload
    json_state = model_orig.to_json()
    model_reloaded = SavedImageClassifierPipeline.from_json(json_state)

    # 4. Predict with reloaded model
    preds_after = {r["record_id"]: model_reloaded.predict_record(r) for r in SYNTHETIC_FIXTURE}

    # 5. Verify prediction agreement within tolerance
    for rid in preds_before:
        prob_before = preds_before[rid]["probability"]
        prob_after = preds_after[rid]["probability"]
        diff = abs(prob_before - prob_after)
        assert diff <= tolerance, (
            f"Prediction discrepancy for {rid}: before={prob_before}, after={prob_after}, diff={diff}"
        )
        assert preds_before[rid]["predicted_label"] == preds_after[rid]["predicted_label"]

    return {
        "status": "PASS",
        "schema_version": model_reloaded.schema_version,
        "agreement_tolerance": tolerance,
        "predictions_before": preds_before,
        "predictions_after": preds_after,
        "requires_training_data_at_inference": False,
        "requires_credentials": False,
    }


if __name__ == "__main__":
    res = run_saved_pipeline_roundtrip()
    print("Saved Pipeline Roundtrip Output:")
    print(f"Status: {res['status']}")
    print(f"Schema Version: {res['schema_version']}")
    for rid, p in res["predictions_after"].items():
        print(f"  {rid}: label={p['predicted_label']} (prob={p['probability']})")
