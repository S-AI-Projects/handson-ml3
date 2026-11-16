"""A small, JSON-serializable text classification pipeline.

This is a deterministic teaching example, not an RNN implementation.
It uses only the Python standard library.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

MODEL_FORMAT = "r33-t020-text-pipeline"
MODEL_VERSION = 1
ALPHA = 1.0


class PipelineError(ValueError):
    """Raised when input data or a saved model violates its contract."""


def tokenize(text):
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def validate_record(record):
    """Validate one prediction record and return its normalized values."""
    if not isinstance(record, Mapping):
        raise PipelineError("record: expected a JSON object")

    if "record_id" not in record:
        raise PipelineError("record_id: required field is missing")

    if "text" not in record:
        raise PipelineError("text: required field is missing")

    if not isinstance(record["record_id"], str) or not record["record_id"].strip():
        raise PipelineError("record_id: expected a non-empty string")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise PipelineError("text: expected a non-empty string")

    unexpected = set(record) - {"record_id", "text"}
    if unexpected:
        field = sorted(unexpected)[0]
        raise PipelineError(f"{field}: unexpected field")

    if not tokenize(record["text"]):
        raise PipelineError("text: must contain at least one alphanumeric token")

    return record["record_id"].strip(), record["text"]


def validate_training_records(records):
    """Validate the training data contract."""
    if not isinstance(records, list) or not records:
        raise PipelineError("records: expected a non-empty list")

    seen = set()

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PipelineError(f"row {index}: expected a JSON object")

        record_id = record.get("record_id")
        text = record.get("text")
        label = record.get("label")

        if not isinstance(record_id, str) or not record_id.strip():
            raise PipelineError(f"row {index}, record_id: expected a non-empty string")

        if record_id in seen:
            raise PipelineError(f"{record_id}, record_id: duplicate ID")
        seen.add(record_id)

        if not isinstance(text, str) or not text.strip() or not tokenize(text):
            raise PipelineError(f"{record_id}, text: expected non-empty tokenizable text")

        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise PipelineError(f"{record_id}, label: expected integer 0 or 1")

        unexpected = set(record) - {"record_id", "text", "label"}
        if unexpected:
            field = sorted(unexpected)[0]
            raise PipelineError(f"{record_id}, {field}: unexpected field")

    if len({record["label"] for record in records}) < 2:
        raise PipelineError("label: training data must contain both classes 0 and 1")


def train_pipeline(records):
    """Fit a deterministic multinomial Naive Bayes text classifier."""
    validate_training_records(records)

    classes = sorted({record["label"] for record in records})
    vocabulary = sorted({
        token
        for record in records
        for token in tokenize(record["text"])
    })

    if not vocabulary:
        raise PipelineError("vocabulary: no tokens available for training")

    class_log_prior = {}
    token_log_prob = {}

    for label in classes:
        class_records = [r for r in records if r["label"] == label]
        class_log_prior[str(label)] = math.log(
            len(class_records) / len(records)
        )

        counts = Counter()
        for record in class_records:
            counts.update(tokenize(record["text"]))

        total_tokens = sum(counts.values())
        denominator = total_tokens + ALPHA * len(vocabulary)

        token_log_prob[str(label)] = {
            token: math.log((counts[token] + ALPHA) / denominator)
            for token in vocabulary
        }

    return {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "config": {
            "algorithm": "multinomial_naive_bayes",
            "tokenizer": "lowercase_alphanumeric",
            "smoothing_alpha": ALPHA,
            "seed": 42,
        },
        "classes": classes,
        "vocabulary": vocabulary,
        "class_log_prior": class_log_prior,
        "token_log_prob": token_log_prob,
    }


def validate_model(model):
    """Check the saved model's format, version, and parameter structure."""
    if not isinstance(model, Mapping):
        raise PipelineError("model: expected a JSON object")

    if model.get("format") != MODEL_FORMAT:
        raise PipelineError("model format mismatch")

    if model.get("version") != MODEL_VERSION:
        raise PipelineError(
            f"unsupported model version: {model.get('version')!r}; "
            f"expected {MODEL_VERSION}"
        )

    required = {
        "config", "classes", "vocabulary",
        "class_log_prior", "token_log_prob"
    }
    missing = required - set(model)
    if missing:
        raise PipelineError(f"model: missing fields {sorted(missing)}")

    classes = model["classes"]
    vocabulary = model["vocabulary"]

    if not isinstance(classes, list) or not classes:
        raise PipelineError("model classes: expected a non-empty list")

    if not isinstance(vocabulary, list) or not vocabulary:
        raise PipelineError("model vocabulary: expected a non-empty list")

    if set(classes) != {0, 1}:
        raise PipelineError("model classes: expected classes 0 and 1")

    priors = model["class_log_prior"]
    probabilities = model["token_log_prob"]

    for label in classes:
        key = str(label)
        if key not in priors or key not in probabilities:
            raise PipelineError(f"model parameters: missing class {key}")

        if not isinstance(priors[key], (int, float)) or not math.isfinite(priors[key]):
            raise PipelineError(f"model prior for class {key}: invalid number")

        if not isinstance(probabilities[key], Mapping):
            raise PipelineError(f"model token probabilities for class {key}: invalid mapping")

        for token in vocabulary:
            value = probabilities[key].get(token)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PipelineError(
                    f"model token probability for class {key}, token {token!r}: invalid number"
                )

    return model


def save_pipeline(model, path):
    """Save a validated model as human-readable JSON."""
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pipeline(path):
    """Load and validate a JSON model. No pickle is used."""
    path = Path(path)

    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"could not load model JSON: {exc}") from exc

    return validate_model(model)


def predict(model, record):
    """Predict one record without access to training data."""
    validate_model(model)
    record_id, text = validate_record(record)
    tokens = tokenize(text)

    scores = {}
    vocabulary = set(model["vocabulary"])

    for label in model["classes"]:
        key = str(label)
        score = model["class_log_prior"][key]

        for token in tokens:
            if token in vocabulary:
                score += model["token_log_prob"][key][token]

        scores[key] = score

    predicted_label = max(
        model["classes"],
        key=lambda label: (scores[str(label)], -label),
    )

    return {
        "record_id": record_id,
        "label": predicted_label,
        "scores": scores,
    }
