"""
R33-T009: Input Perturbation Robustness for Gradient Boosted Classifier
Chapter 07: Ensemble Learning and Random Forests / Gradient Boosting

Contract:
- Test a stated robustness property for the classification interface:
  record-order changes with stable IDs (permutation invariance).
- One valid transformation preserves the documented result after ID alignment.
- One unsupported input (incompatible shape / dimensionality violation) is rejected.
- The test does NOT claim broad adversarial robustness.
- Eight synthetic rows with x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1].
- Fixed stratified split: train (4 rows: 2 pos, 2 neg), val (2 rows: 1 pos, 1 neg), test (2 rows: 1 pos, 1 neg).
- Offline, CPU-only, standard-library implementation with zero external dependencies.
- Labelled strictly as pipeline verification on synthetic data, NOT real-world performance evidence.
"""

from __future__ import annotations
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


class IncompatibleShapeError(Exception):
    """Raised when an input row has an invalid number of features."""
    pass


class InvalidFeatureValueError(Exception):
    """Raised when an input feature contains NaN, Inf, or unsupported types."""
    pass


@dataclass
class DatasetRecord:
    record_id: str
    group_id: str
    x1: float
    x2: float
    label: int
    split: str

    def to_features(self) -> List[float]:
        return [self.x1, self.x2]


@dataclass
class DecisionStump:
    feature_idx: int
    threshold: float
    val_left: float
    val_right: float

    def predict_val(self, x: List[float]) -> float:
        if x[self.feature_idx] <= self.threshold:
            return self.val_left
        return self.val_right


class GradientBoostedStumpClassifier:
    """
    Gradient Boosted Decision Stump Classifier for binary classification.
    Uses logit link: p = 1 / (1 + exp(-F(x))).
    Fits sequentially on pseudo-residuals (negative gradient of binary cross-entropy).
    Splits are chosen by minimizing residual sum of squares (variance reduction),
    followed by Newton-Raphson leaf step updates.
    """

    def __init__(self, n_estimators: int = 3, learning_rate: float = 0.5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_log_odds: float = 0.0
        self.stumps: List[DecisionStump] = []
        self.is_fitted: bool = False

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 30.0:
            return 1.0
        if z <= -30.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, records: List[DatasetRecord]) -> None:
        N = len(records)
        if N == 0:
            raise ValueError("Cannot fit on empty dataset.")

        y = [r.label for r in records]
        X = [r.to_features() for r in records]

        # Calculate initial log odds F_0
        pos_count = sum(y)
        p_prior = max(1e-6, min(1.0 - 1e-6, pos_count / N))
        self.initial_log_odds = math.log(p_prior / (1.0 - p_prior))

        F = [self.initial_log_odds] * N
        self.stumps = []

        # Candidate threshold pool for feature x1 and x2
        x1_vals = sorted(list(set(row[0] for row in X)))
        candidate_thresh_x1 = [
            (x1_vals[i] + x1_vals[i + 1]) / 2.0 for i in range(len(x1_vals) - 1)
        ]
        # Also include standard unit half-steps if needed
        candidate_thresh_x1.extend([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        candidate_thresh_x1 = sorted(list(set(candidate_thresh_x1)))

        for _ in range(self.n_estimators):
            # Compute probabilities and negative gradients (residuals)
            probs = [self._sigmoid(f_val) for f_val in F]
            residuals = [y[i] - probs[i] for i in range(N)]

            best_rss = float("inf")
            best_stump = None

            for feat_idx in [0, 1]:
                thresholds = candidate_thresh_x1 if feat_idx == 0 else [0.5, 1.0, 1.5]
                for thresh in thresholds:
                    left_indices = [i for i in range(N) if X[i][feat_idx] <= thresh]
                    right_indices = [i for i in range(N) if X[i][feat_idx] > thresh]

                    if not left_indices or not right_indices:
                        continue

                    # Variance reduction / RSS on pseudo-residuals
                    mean_left_res = sum(residuals[i] for i in left_indices) / len(left_indices)
                    mean_right_res = sum(residuals[i] for i in right_indices) / len(right_indices)

                    rss = sum(
                        (residuals[i] - mean_left_res) ** 2 for i in left_indices
                    ) + sum(
                        (residuals[i] - mean_right_res) ** 2 for i in right_indices
                    )

                    if rss < best_rss:
                        # Newton-Raphson leaf updates
                        num_left = sum(residuals[i] for i in left_indices)
                        denom_left = sum(probs[i] * (1.0 - probs[i]) for i in left_indices)
                        val_left = num_left / max(1e-6, denom_left)

                        num_right = sum(residuals[i] for i in right_indices)
                        denom_right = sum(probs[i] * (1.0 - probs[i]) for i in right_indices)
                        val_right = num_right / max(1e-6, denom_right)

                        best_rss = rss
                        best_stump = DecisionStump(
                            feature_idx=feat_idx,
                            threshold=thresh,
                            val_left=val_left,
                            val_right=val_right,
                        )

            if best_stump is not None:
                self.stumps.append(best_stump)
                for i in range(N):
                    F[i] += self.learning_rate * best_stump.predict_val(X[i])

        self.is_fitted = True

    def validate_features(self, x: List[float]) -> None:
        """Validate feature count and finite numerical values."""
        if len(x) != 2:
            raise IncompatibleShapeError(
                f"Incompatible input shape: Expected 2 features ([x1, x2]), got {len(x)}."
            )
        for val in x:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise InvalidFeatureValueError(
                    f"Invalid feature value '{val}': Must be finite float or int."
                )

    def predict_record(self, record: DatasetRecord) -> Tuple[int, float]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        feats = record.to_features()
        self.validate_features(feats)

        F_val = self.initial_log_odds
        for stump in self.stumps:
            F_val += self.learning_rate * stump.predict_val(feats)

        prob = self._sigmoid(F_val)
        pred_label = 1 if prob >= 0.5 else 0
        return pred_label, prob

    def predict_batch(self, records: List[DatasetRecord]) -> Dict[str, int]:
        """Returns mapping from record_id to predicted binary class."""
        predictions = {}
        for r in records:
            label, _ = self.predict_record(r)
            predictions[r.record_id] = label
        return predictions

    def predict_proba_batch(self, records: List[DatasetRecord]) -> Dict[str, float]:
        """Returns mapping from record_id to predicted probability of class 1."""
        probas = {}
        for r in records:
            _, prob = self.predict_record(r)
            probas[r.record_id] = prob
        return probas


def load_dataset(filepath: Optional[str] = None, format: str = "json") -> List[DatasetRecord]:
    """Load the 8-row synthetic classification fixture from JSON or CSV."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "fixture.json" if format.lower() == "json" else "fixture.csv"
        filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fixture file not found: {filepath}")

    records: List[DatasetRecord] = []

    if format.lower() == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["records"]:
            record = DatasetRecord(
                record_id=item["record_id"],
                group_id=item["group_id"],
                x1=float(item["x1"]),
                x2=float(item["x2"]),
                label=int(item["label"]),
                split=item.get("split", "train"),
            )
            records.append(record)

    elif format.lower() == "csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = DatasetRecord(
                    record_id=row["record_id"],
                    group_id=row["group_id"],
                    x1=float(row["x1"]),
                    x2=float(row["x2"]),
                    label=int(row["label"]),
                    split=row.get("split", "train"),
                )
                records.append(record)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if len(records) != 8:
        raise ValueError(f"Expected 8 fixture records, got {len(records)}")

    return records


def permute_record_order(
    records: List[DatasetRecord], seed: int = 42
) -> List[DatasetRecord]:
    """
    Valid transformation: Permutes the sequence of records in a batch.
    Preserves all internal record attributes and stable IDs.
    """
    rng = random.Random(seed)
    permuted = list(records)
    rng.shuffle(permuted)
    return permuted


@dataclass
class RobustnessVerificationResult:
    transformation_name: str
    order_invariance_verified: bool
    unsupported_shape_rejected: bool
    unsupported_value_rejected: bool
    num_records_tested: int
    predictions_original: Dict[str, int]
    predictions_permuted: Dict[str, int]
    accuracy_original: float
    accuracy_permuted: float
    is_synthetic: bool = True
    claims_broad_adversarial_robustness: bool = False
    cpu_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def test_robustness_properties(
    model: GradientBoostedStumpClassifier,
    records: List[DatasetRecord],
    permutation_seed: int = 42,
) -> RobustnessVerificationResult:
    """
    Execute robustness verification:
    1. Valid transformation: Record order permutation preserves predictions per ID.
    2. Unsupported input: Shape mismatch raises IncompatibleShapeError.
    3. Unsupported value: NaN/Inf raises InvalidFeatureValueError.
    """
    # 1. Original predictions
    preds_orig = model.predict_batch(records)
    probas_orig = model.predict_proba_batch(records)

    # 2. Permuted predictions
    permuted = permute_record_order(records, seed=permutation_seed)
    preds_perm = model.predict_batch(permuted)
    probas_perm = model.predict_proba_batch(permuted)

    # Check ID alignment invariance
    order_invariant = True
    for r in records:
        if preds_orig[r.record_id] != preds_perm[r.record_id]:
            order_invariant = False
        if abs(probas_orig[r.record_id] - probas_perm[r.record_id]) > 1e-9:
            order_invariant = False

    # Accuracy calculations
    acc_orig = sum(1 for r in records if preds_orig[r.record_id] == r.label) / len(records)
    acc_perm = sum(1 for r in records if preds_perm[r.record_id] == r.label) / len(records)

    # 3. Test unsupported shape rejection
    shape_rejected = False
    try:
        model.validate_features([1.0, 2.0, 3.0])  # 3 features instead of 2
    except IncompatibleShapeError:
        shape_rejected = True

    # 4. Test unsupported NaN/Inf value rejection
    val_rejected = False
    try:
        model.validate_features([float("nan"), 1.0])
    except InvalidFeatureValueError:
        val_rejected = True

    return RobustnessVerificationResult(
        transformation_name="record_order_permutation_with_stable_ids",
        order_invariance_verified=order_invariant,
        unsupported_shape_rejected=shape_rejected,
        unsupported_value_rejected=val_rejected,
        num_records_tested=len(records),
        predictions_original=preds_orig,
        predictions_permuted=preds_perm,
        accuracy_original=acc_orig,
        accuracy_permuted=acc_perm,
        is_synthetic=True,
        claims_broad_adversarial_robustness=False,
        cpu_only=True,
    )
