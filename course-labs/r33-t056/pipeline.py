"""Tiny saved-pipeline interface: nearest-centroid classifier on 4x4 images.

Persistence is plain JSON only. Loading never executes code and never
touches untrusted binary formats. Inference needs only the saved file:
no training data, credentials or network.

Saved document layout:
{
  "schema_version": 1,
  "model_type": "nearest_centroid",
  "config": {"image_size": 4, "pixel_max": 255},
  "params": {"centroids": {"0": [16 floats], "1": [16 floats]}}
}
"""
import json
import math

SCHEMA_VERSION = 1
MODEL_TYPE = "nearest_centroid"
IMAGE_SIZE = 4
PIXEL_MAX = 255
CONFIG = {"image_size": IMAGE_SIZE, "pixel_max": PIXEL_MAX}
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE


class PipelineSchemaError(ValueError):
    """Raised for any schema, version, shape or value problem."""


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _flatten(pixels):
    if not isinstance(pixels, list) or len(pixels) != IMAGE_SIZE:
        raise PipelineSchemaError(f"pixels must be a list of {IMAGE_SIZE} rows")
    flat = []
    for row in pixels:
        if not isinstance(row, list) or len(row) != IMAGE_SIZE:
            raise PipelineSchemaError(f"each row must have {IMAGE_SIZE} values")
        for v in row:
            if not _is_number(v) or not 0 <= v <= PIXEL_MAX:
                raise PipelineSchemaError(f"pixel values must be finite numbers in 0..{PIXEL_MAX}")
            flat.append(v / PIXEL_MAX)
    return flat


def validate_model(model):
    if not isinstance(model, dict):
        raise PipelineSchemaError("model document must be a JSON object")
    for key in ("schema_version", "model_type", "config", "params"):
        if key not in model:
            raise PipelineSchemaError(f"missing key: {key}")
    if model["schema_version"] != SCHEMA_VERSION:
        raise PipelineSchemaError(
            f"unsupported schema_version {model['schema_version']!r}; expected {SCHEMA_VERSION}")
    if model["model_type"] != MODEL_TYPE:
        raise PipelineSchemaError(
            f"unsupported model_type {model['model_type']!r}; expected {MODEL_TYPE!r}")
    if model["config"] != CONFIG:
        raise PipelineSchemaError(f"config mismatch; expected {CONFIG}")
    params = model["params"]
    if not isinstance(params, dict) or not isinstance(params.get("centroids"), dict):
        raise PipelineSchemaError("params.centroids must be an object")
    cents = params["centroids"]
    if set(cents) != {"0", "1"}:
        raise PipelineSchemaError("centroids must have exactly the keys '0' and '1'")
    for label, vec in cents.items():
        if not isinstance(vec, list) or len(vec) != N_FEATURES:
            raise PipelineSchemaError(f"centroid '{label}' must have {N_FEATURES} values")
        if not all(_is_number(v) for v in vec):
            raise PipelineSchemaError(f"centroid '{label}' has a non-finite or non-numeric value")
    return model


def fit(records):
    """Learn per-class mean images (scaled to 0..1) from TRAINING records."""
    sums = {0: [0.0] * N_FEATURES, 1: [0.0] * N_FEATURES}
    counts = {0: 0, 1: 0}
    for r in records:
        if r["label"] not in sums:
            raise PipelineSchemaError("labels must be 0 or 1")
        flat = _flatten(r["pixels"])
        sums[r["label"]] = [s + v for s, v in zip(sums[r["label"]], flat)]
        counts[r["label"]] += 1
    if counts[0] == 0 or counts[1] == 0:
        raise PipelineSchemaError("training data must contain both classes")
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "config": dict(CONFIG),
        "params": {"centroids": {str(k): [s / counts[k] for s in sums[k]] for k in (0, 1)}},
    }


def save(model, path):
    validate_model(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=True)


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
    except ValueError as exc:          # JSONDecodeError and UnicodeDecodeError
        raise PipelineSchemaError(f"not a valid JSON model file: {exc}") from exc
    return validate_model(model)


def predict(model, pixels):
    """Return {'label': int, 'scores': {'0': dist, '1': dist}} (squared distances)."""
    validate_model(model)
    flat = _flatten(pixels)
    scores = {k: sum((a - b) ** 2 for a, b in zip(flat, vec))
              for k, vec in model["params"]["centroids"].items()}
    label = min((0, 1), key=lambda c: (scores[str(c)], c))   # ties -> smaller label
    return {"label": label, "scores": scores}
