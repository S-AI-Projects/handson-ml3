"""
R33-T037: Record reproducible configuration
===========================================
Project/section: 15 processing sequences using rnns and cnns / LSTMs
Notebook: 15_processing_sequences_using_rnns_and_cnns.ipynb

Goal
----
Write a serializable configuration for the time series fixture with seed,
schema version, feature ordering, split IDs and algorithm settings; reject
unknown or contradictory settings.

Fixture:
Ten consecutive daily observations with values [10,12,11,13,15,14,16,18,17,19]
and IDs d1..d10. The first six are training, next two validation and final two test;
no shuffling.

Data contract:
- Source: Fully synthetic, generated for configuration reproducibility verification only.
- Licence: N/A.
- Real dataset used: No.
- Disclaimer: These records are simulated. Results must not be cited as evidence
  of real-world or clinical performance.
"""

import copy
import json
import math
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Synthetic Fixture
# ---------------------------------------------------------------------------

SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "d1", "timestamp": "2026-01-01", "value": 10.0},
    {"record_id": "d2", "timestamp": "2026-01-02", "value": 12.0},
    {"record_id": "d3", "timestamp": "2026-01-03", "value": 11.0},
    {"record_id": "d4", "timestamp": "2026-01-04", "value": 13.0},
    {"record_id": "d5", "timestamp": "2026-01-05", "value": 15.0},
    {"record_id": "d6", "timestamp": "2026-01-06", "value": 14.0},
    {"record_id": "d7", "timestamp": "2026-01-07", "value": 16.0},
    {"record_id": "d8", "timestamp": "2026-01-08", "value": 18.0},
    {"record_id": "d9", "timestamp": "2026-01-09", "value": 17.0},
    {"record_id": "d10", "timestamp": "2026-01-10", "value": 19.0},
]

EXPECTED_TRAIN_IDS = ["d1", "d2", "d3", "d4", "d5", "d6"]
EXPECTED_VAL_IDS = ["d7", "d8"]
EXPECTED_TEST_IDS = ["d9", "d10"]

ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "seed",
    "feature_ordering",
    "split_ids",
    "algorithm_settings",
}

ALLOWED_ALGORITHM_KEYS = {
    "model_type",
    "lag",
    "clip_min",
    "clip_max",
    "learning_rate",
}

ALLOWED_SPLIT_KEYS = {"train", "validation", "test"}


# ---------------------------------------------------------------------------
# Configuration Class
# ---------------------------------------------------------------------------

class PipelineConfig:
    """
    Serializable, strictly validated configuration for sequence modeling.
    Enforces schema versioning, strict key whitelisting, split isolation,
    and parameter validity.
    """

    def __init__(
        self,
        schema_version: str = "1.0.0",
        seed: int = 42,
        feature_ordering: List[str] = None,
        split_ids: Dict[str, List[str]] = None,
        algorithm_settings: Dict[str, Any] = None,
    ):
        self.schema_version = schema_version
        self.seed = seed
        self.feature_ordering = (
            feature_ordering if feature_ordering is not None else ["value"]
        )
        self.split_ids = (
            split_ids
            if split_ids is not None
            else {
                "train": list(EXPECTED_TRAIN_IDS),
                "validation": list(EXPECTED_VAL_IDS),
                "test": list(EXPECTED_TEST_IDS),
            }
        )
        self.algorithm_settings = (
            algorithm_settings
            if algorithm_settings is not None
            else {
                "model_type": "autoregressive_lag",
                "lag": 1,
                "clip_min": 0.0,
                "clip_max": 100.0,
                "learning_rate": 0.01,
            }
        )
        self.validate()

    def validate(self) -> None:
        """Strictly validates configuration integrity, rejecting unknown or contradictory keys."""
        # 1. Schema version
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string.")
        if self.schema_version != "1.0.0":
            raise ValueError(f"Unsupported schema_version '{self.schema_version}'. Expected '1.0.0'.")

        # 2. Seed
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        # 3. Feature ordering
        if not isinstance(self.feature_ordering, list) or not self.feature_ordering:
            raise ValueError("feature_ordering must be a non-empty list of feature names.")
        if any(not isinstance(f, str) or not f.strip() for f in self.feature_ordering):
            raise ValueError("All feature_ordering items must be valid non-empty strings.")

        # 4. Split IDs
        if not isinstance(self.split_ids, dict):
            raise ValueError("split_ids must be a dictionary.")
        unknown_splits = set(self.split_ids.keys()) - ALLOWED_SPLIT_KEYS
        if unknown_splits:
            raise ValueError(f"Unknown split keys: {unknown_splits}")
        missing_splits = ALLOWED_SPLIT_KEYS - set(self.split_ids.keys())
        if missing_splits:
            raise ValueError(f"Missing required split keys: {missing_splits}")

        train_set = set(self.split_ids["train"])
        val_set = set(self.split_ids["validation"])
        test_set = set(self.split_ids["test"])

        if not train_set or not val_set or not test_set:
            raise ValueError("Train, validation, and test splits must each be non-empty.")

        # Check disjointness (no split leakage / contradictory overlap)
        if not train_set.isdisjoint(val_set):
            raise ValueError("Contradictory split: Overlap found between train and validation sets.")
        if not train_set.isdisjoint(test_set):
            raise ValueError("Contradictory split: Overlap found between train and test sets.")
        if not val_set.isdisjoint(test_set):
            raise ValueError("Contradictory split: Overlap found between validation and test sets.")

        # 5. Algorithm settings
        if not isinstance(self.algorithm_settings, dict):
            raise ValueError("algorithm_settings must be a dictionary.")
        unknown_algo_keys = set(self.algorithm_settings.keys()) - ALLOWED_ALGORITHM_KEYS
        if unknown_algo_keys:
            raise ValueError(f"Unknown algorithm_settings keys: {unknown_algo_keys}")

        lag = self.algorithm_settings.get("lag", 1)
        if not isinstance(lag, int) or lag < 1:
            raise ValueError("algorithm_settings['lag'] must be an integer >= 1.")
        if lag >= len(self.split_ids["train"]):
            raise ValueError(
                f"algorithm_settings['lag'] ({lag}) cannot be >= training set size ({len(self.split_ids['train'])})."
            )

        clip_min = self.algorithm_settings.get("clip_min", 0.0)
        clip_max = self.algorithm_settings.get("clip_max", 100.0)
        if clip_min >= clip_max:
            raise ValueError("clip_min must be strictly less than clip_max.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "feature_ordering": list(self.feature_ordering),
            "split_ids": {k: list(v) for k, v in self.split_ids.items()},
            "algorithm_settings": copy.deepcopy(self.algorithm_settings),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Reconstruct PipelineConfig from dictionary, strictly rejecting unknown fields."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        unknown_keys = set(data.keys()) - ALLOWED_TOP_LEVEL_KEYS
        if unknown_keys:
            raise ValueError(f"Unknown top-level configuration keys: {unknown_keys}")
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            seed=data.get("seed", 42),
            feature_ordering=data.get("feature_ordering"),
            split_ids=data.get("split_ids"),
            algorithm_settings=data.get("algorithm_settings"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineConfig":
        """Deserialize configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Sequence Model & Verification Pipeline
# ---------------------------------------------------------------------------

class AutoregressiveSequenceModel:
    """Lightweight CPU autoregressive sequence baseline using configured lag."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.lag = config.algorithm_settings.get("lag", 1)
        self.clip_min = config.algorithm_settings.get("clip_min", 0.0)
        self.clip_max = config.algorithm_settings.get("clip_max", 100.0)
        self.weight: float = 1.0
        self.bias: float = 0.0
        self.fitted = False

    def _validate_input_records(self, records: List[Dict[str, Any]]) -> None:
        """Validate input records schema and dimensions."""
        if not records:
            raise ValueError("Input records cannot be empty.")
        for r in records:
            if "record_id" not in r:
                raise ValueError("Record missing required 'record_id' field.")
            for feat in self.config.feature_ordering:
                if feat not in r:
                    raise ValueError(f"Incompatible input: Missing required feature '{feat}'.")
                if not isinstance(r[feat], (int, float)) or not math.isfinite(r[feat]):
                    raise ValueError(f"Incompatible input: Feature '{feat}' must be finite numeric.")

    def fit(self, records: List[Dict[str, Any]]) -> "AutoregressiveSequenceModel":
        self._validate_input_records(records)
        values = [float(r[self.config.feature_ordering[0]]) for r in records]
        if len(values) <= self.lag:
            raise ValueError(f"Records length ({len(values)}) must be > lag ({self.lag}).")

        # Compute simple least-squares autoregressive coefficient on train sequence
        xs = values[:-1]
        ys = values[1:]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 1e-9:
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            self.weight = cov / var_x
            self.bias = mean_y - self.weight * mean_x
        else:
            self.weight = 1.0
            self.bias = 0.0

        self.fitted = True
        return self

    def predict_next(self, prev_values: List[float]) -> float:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict_next().")
        if len(prev_values) < self.lag:
            raise ValueError(f"Need at least {self.lag} previous values.")
        val = self.weight * prev_values[-1] + self.bias
        return max(self.clip_min, min(self.clip_max, val))

    def evaluate_split(
        self, context_records: List[Dict[str, Any]], target_records: List[Dict[str, Any]]
    ) -> List[float]:
        """Predicts sequential values for target_records using context history."""
        self._validate_input_records(context_records + target_records)
        history = [float(r[self.config.feature_ordering[0]]) for r in context_records]
        preds = []
        for r in target_records:
            pred = self.predict_next(history)
            preds.append(round(pred, 4))
            history.append(float(r[self.config.feature_ordering[0]]))
        return preds


def run_pipeline(config: PipelineConfig = None) -> Dict[str, Any]:
    """Runs data partitioning, model fitting, and validation according to config."""
    if config is None:
        config = PipelineConfig()

    id_map = {r["record_id"]: r for r in SYNTHETIC_FIXTURE}
    train_records = [id_map[rid] for rid in config.split_ids["train"]]
    val_records = [id_map[rid] for rid in config.split_ids["validation"]]
    test_records = [id_map[rid] for rid in config.split_ids["test"]]

    model = AutoregressiveSequenceModel(config)
    model.fit(train_records)

    val_preds = model.evaluate_split(train_records, val_records)
    test_preds = model.evaluate_split(train_records + val_records, test_records)

    return {
        "config": config.to_dict(),
        "train_ids": config.split_ids["train"],
        "val_ids": config.split_ids["validation"],
        "test_ids": config.split_ids["test"],
        "val_predictions": val_preds,
        "test_predictions": test_preds,
        "fitted_weight": round(model.weight, 4),
        "fitted_bias": round(model.bias, 4),
    }


if __name__ == "__main__":
    cfg = PipelineConfig()
    result = run_pipeline(cfg)
    print("Configuration Pipeline Output:")
    print(json.dumps(result, indent=2))
