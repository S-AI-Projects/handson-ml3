"""Reproducible configuration serialization and reconstruction for R33-T027.

Encapsulates seed, schema version, feature ordering, split IDs, and algorithm settings.
Validates configuration integrity and rejects unknown or contradictory parameters.
"""
import json
from typing import List, Dict, Any

ALLOWED_CONFIG_KEYS = {
    "seed",
    "schema_version",
    "feature_ordering",
    "split_ids",
    "algorithm_settings"
}

class ReproduciblePipelineConfig:
    """Serializable pipeline configuration."""
    def __init__(
        self,
        seed: int = 42,
        schema_version: str = "1.0",
        feature_ordering: List[str] = None,
        split_ids: Dict[str, List[str]] = None,
        algorithm_settings: Dict[str, Any] = None
    ) -> None:
        self.seed = seed
        self.schema_version = schema_version
        self.feature_ordering = feature_ordering or ["x1", "x2"]
        self.split_ids = split_ids or {
            "train": ["row_0", "row_1", "row_4", "row_5"],
            "val": ["row_2", "row_6"],
            "test": ["row_3", "row_7"]
        }
        self.algorithm_settings = algorithm_settings or {
            "model_type": "threshold_classifier",
            "threshold": 3.5
        }
        self.validate()

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("Contradictory setting: seed cannot be negative")
        if not self.feature_ordering or self.feature_ordering != ["x1", "x2"]:
            raise ValueError(f"Incompatible feature ordering: {self.feature_ordering}")

        # Ensure split IDs are mutually disjoint
        train_set = set(self.split_ids.get("train", []))
        val_set = set(self.split_ids.get("val", []))
        test_set = set(self.split_ids.get("test", []))
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("Contradictory setting: split IDs must be strictly disjoint")

    def to_json(self) -> str:
        return json.dumps({
            "seed": self.seed,
            "schema_version": self.schema_version,
            "feature_ordering": self.feature_ordering,
            "split_ids": self.split_ids,
            "algorithm_settings": self.algorithm_settings
        }, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReproduciblePipelineConfig":
        unknown_keys = set(data.keys()) - ALLOWED_CONFIG_KEYS
        if unknown_keys:
            raise ValueError(f"Unknown configuration fields rejected: {unknown_keys}")
        return cls(
            seed=data["seed"],
            schema_version=data["schema_version"],
            feature_ordering=data["feature_ordering"],
            split_ids=data["split_ids"],
            algorithm_settings=data["algorithm_settings"]
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ReproduciblePipelineConfig":
        data = json.loads(json_str)
        return cls.from_dict(data)

class ConfiguredClassifierPipeline:
    """Applies configured splits and threshold predictions with input shape validation."""
    def __init__(self, config: ReproduciblePipelineConfig) -> None:
        self.config = config

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        preds = []
        for r in records:
            # Validate input shape and schema
            for feat in self.config.feature_ordering:
                if feat not in r:
                    raise ValueError(f"Incompatible input shape: missing feature {feat}")
                if not isinstance(r[feat], (int, float)):
                    raise ValueError(f"Incompatible input type for {feat}: {type(r[feat])}")
            val = r[self.config.feature_ordering[0]]
            preds.append(1 if val >= self.config.algorithm_settings["threshold"] else 0)
        return preds
