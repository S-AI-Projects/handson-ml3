"""
Course Lab R33-T046: Serializable Configuration Schema
Defines a robust, serializable experiment configuration for decision tree classification fixtures.
"""
import json
from typing import Dict, Any, List

ALLOWED_CRITERIA = ("gini", "entropy", "log_loss")
ALLOWED_FEATURES = ("x1", "x2")

class DecisionTreeConfig:
    def __init__(self, schema_version: str, random_seed: int, feature_ordering: List[str], split_ids: Dict[str, List[str]], algorithm_settings: Dict[str, Any]):
        self.schema_version = schema_version
        self.random_seed = random_seed
        self.feature_ordering = list(feature_ordering)
        self.split_ids = {k: list(v) for k, v in split_ids.items()}
        self.algorithm_settings = dict(algorithm_settings)
        self.validate()

    def validate(self):
        if not self.schema_version or not isinstance(self.schema_version, str):
            raise ValueError("Invalid schema_version")

        if not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")

        if not self.feature_ordering or set(self.feature_ordering) != set(ALLOWED_FEATURES):
            raise ValueError(f"feature_ordering must match supported features {ALLOWED_FEATURES}")

        required_splits = ("train", "validation", "test")
        if not all(s in self.split_ids for s in required_splits):
            raise ValueError(f"split_ids must contain all of {required_splits}")

        # Check split disjointness
        train_s = set(self.split_ids["train"])
        val_s = set(self.split_ids["validation"])
        test_s = set(self.split_ids["test"])
        if train_s.intersection(val_s) or train_s.intersection(test_s) or val_s.intersection(test_s):
            raise ValueError("Split ID sets must be completely disjoint")

        # Algorithm settings validation
        depth = self.algorithm_settings.get("max_depth")
        if depth is not None and (not isinstance(depth, int) or depth <= 0):
            raise ValueError("max_depth must be a positive integer")

        crit = self.algorithm_settings.get("criterion", "gini")
        if crit not in ALLOWED_CRITERIA:
            raise ValueError(f"criterion must be one of {ALLOWED_CRITERIA}")

        min_samples = self.algorithm_settings.get("min_samples_split", 2)
        if not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError("min_samples_split must be an integer >= 2")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "random_seed": self.random_seed,
            "feature_ordering": self.feature_ordering,
            "split_ids": self.split_ids,
            "algorithm_settings": self.algorithm_settings
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionTreeConfig":
        allowed_keys = {"schema_version", "random_seed", "feature_ordering", "split_ids", "algorithm_settings"}
        extra_keys = set(data.keys()) - allowed_keys
        if extra_keys:
            raise ValueError(f"Unknown configuration keys rejected: {extra_keys}")

        return cls(
            schema_version=data["schema_version"],
            random_seed=data["random_seed"],
            feature_ordering=data["feature_ordering"],
            split_ids=data["split_ids"],
            algorithm_settings=data["algorithm_settings"]
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DecisionTreeConfig":
        return cls.from_dict(json.loads(json_str))
