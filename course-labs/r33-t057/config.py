"""Serializable, validated configuration for the synthetic image fixture.

Config document (plain JSON, never pickle):
{
  "schema_version": 1,
  "seed": 0,
  "input_shape": [4, 4],
  "feature_order": ["p00", "p01", ..., "p33"],      # row-major pixel names
  "split_ids": {"train": [...], "val": [...], "test": [...]},   # record IDs
  "algorithm": {"name": "nearest_centroid", "pixel_max": 255}
}
Unknown or contradictory settings raise ConfigError.
"""
import json
import platform
import random

SCHEMA_VERSION = 1
IMAGE_SHAPE = [4, 4]
PIXEL_MAX = 255
ALGORITHM = "nearest_centroid"
FEATURE_ORDER = [f"p{r}{c}" for r in range(4) for c in range(4)]
TOP_KEYS = {"schema_version", "seed", "input_shape", "feature_order", "split_ids", "algorithm"}
SPLIT_NAMES = ("train", "val", "test")


class ConfigError(ValueError):
    """Raised for any unknown, missing, incompatible or contradictory setting."""


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def validate_config(cfg, records=None):
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a JSON object")
    extra = set(cfg) - TOP_KEYS
    if extra:
        raise ConfigError(f"unknown field(s): {sorted(extra)}")
    missing = TOP_KEYS - set(cfg)
    if missing:
        raise ConfigError(f"missing field(s): {sorted(missing)}")
    if not _is_int(cfg["schema_version"]) or cfg["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"unsupported schema_version {cfg['schema_version']!r}; expected {SCHEMA_VERSION}")
    if not _is_int(cfg["seed"]) or cfg["seed"] < 0:
        raise ConfigError("seed must be a non-negative integer")
    if cfg["input_shape"] != IMAGE_SHAPE:
        raise ConfigError(f"input_shape must be {IMAGE_SHAPE}")
    fo = cfg["feature_order"]
    if (not isinstance(fo, list) or not all(isinstance(x, str) for x in fo)
            or sorted(fo) != sorted(FEATURE_ORDER)):
        raise ConfigError("feature_order must list each of the 16 pixel names exactly once")
    split = cfg["split_ids"]
    if not isinstance(split, dict) or set(split) != set(SPLIT_NAMES):
        raise ConfigError("split_ids must have exactly the keys train, val, test")
    seen = {}
    for name in SPLIT_NAMES:
        ids = split[name]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise ConfigError(f"split_ids.{name} must be a non-empty list of record IDs")
        for i in ids:
            if i in seen:
                raise ConfigError(f"record id {i} appears in more than one split")
            seen[i] = name
    alg = cfg["algorithm"]
    if not isinstance(alg, dict):
        raise ConfigError("algorithm must be an object")
    extra = set(alg) - {"name", "pixel_max"}
    if extra:
        raise ConfigError(f"unknown field(s) in algorithm: {sorted(extra)}")
    missing = {"name", "pixel_max"} - set(alg)
    if missing:
        raise ConfigError(f"missing field(s) in algorithm: {sorted(missing)}")
    if alg["name"] != ALGORITHM:
        raise ConfigError(f"algorithm.name must be {ALGORITHM!r}")
    if not _is_int(alg["pixel_max"]) or alg["pixel_max"] != PIXEL_MAX:
        raise ConfigError(f"algorithm.pixel_max must be {PIXEL_MAX}")
    if records is not None:
        by_id = {r["record_id"]: r for r in records}
        for i in seen:
            if i not in by_id:
                raise ConfigError(f"unknown record id: {i}")
        group_split = {}
        for i, name in seen.items():
            g = by_id[i]["group_id"]
            if group_split.setdefault(g, name) != name:
                raise ConfigError(f"group leakage: group {g} appears in more than one split")
        if {by_id[i]["label"] for i in split["train"]} != {0, 1}:
            raise ConfigError("train split must contain both classes")
    return cfg


def make_config(records, seed=0):
    """Build a config; the seed deterministically assigns whole groups to splits."""
    if not _is_int(seed) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")
    groups = sorted({r["group_id"] for r in records})
    if len(groups) != 3:
        raise ConfigError("expected exactly 3 groups")
    random.Random(seed).shuffle(groups)
    split = {name: [r["record_id"] for r in records if r["group_id"] == g]
             for name, g in zip(SPLIT_NAMES, groups)}
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "input_shape": list(IMAGE_SHAPE),
        "feature_order": list(FEATURE_ORDER),
        "split_ids": split,
        "algorithm": {"name": ALGORITHM, "pixel_max": PIXEL_MAX},
    }
    return validate_config(cfg, records)


def save_config(cfg, path):
    validate_config(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def load_config(path, records=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except ValueError as exc:
        raise ConfigError(f"not a valid JSON config file: {exc}") from exc
    return validate_config(cfg, records)


def splits_from_config(cfg, records):
    validate_config(cfg, records)
    by_id = {r["record_id"]: r for r in records}
    return {name: [by_id[i] for i in cfg["split_ids"][name]] for name in SPLIT_NAMES}


def extract_features(cfg, pixels):
    """Scaled features in the config's feature_order; rejects incompatible shapes."""
    if not isinstance(pixels, list) or len(pixels) != IMAGE_SHAPE[0]:
        raise ConfigError(f"input must have shape {IMAGE_SHAPE}")
    for row in pixels:
        if not isinstance(row, list) or len(row) != IMAGE_SHAPE[1]:
            raise ConfigError(f"input must have shape {IMAGE_SHAPE}")
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= PIXEL_MAX:
                raise ConfigError(f"pixel values must be numbers in 0..{PIXEL_MAX}")
    return [pixels[int(n[1])][int(n[2])] / cfg["algorithm"]["pixel_max"] for n in cfg["feature_order"]]


def fit(cfg, records):
    """Per-class centroids from the config's TRAIN split only."""
    train = splits_from_config(cfg, records)["train"]
    cents = {}
    for label in (0, 1):
        feats = [extract_features(cfg, r["pixels"]) for r in train if r["label"] == label]
        cents[str(label)] = [sum(col) / len(feats) for col in zip(*feats)]
    return cents


def predict(cfg, centroids, pixels):
    feats = extract_features(cfg, pixels)
    scores = {k: sum((a - b) ** 2 for a, b in zip(feats, vec)) for k, vec in centroids.items()}
    label = min((0, 1), key=lambda c: (scores[str(c)], c))
    return {"label": label, "scores": scores}


def environment_info():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
