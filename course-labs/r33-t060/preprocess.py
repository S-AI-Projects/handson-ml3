"""Explicit preprocessing for a TF-Serving-style REST query (offline).

Contract
  input : 4x4 nested list of ints in 0..255 (rows of pixels, no channel axis)
  output: Float32Image, dtype float32, shape (4, 4, 1) - explicit channel axis,
          values pixel / 255.0 so the range is 0.0 .. 1.0
Rejected: wrong dimensions (not exactly 4x4 rows, ragged, already 3-D, flat),
          out-of-range pixels (<0 or >255), non-integer pixels (floats, bools,
          strings, None).

Leakage control: the only fitting entry point is Scaler.fit(), and it accepts
training-split records only.  transform() is pure and never changes fitted
state, so held-out input cannot be fitted through the public interface.
The 0..255 -> 0..1 scaling itself is a fixed-range transform (no learned
statistics); fit() records which training records the scaler was fitted on.
"""
import json
from array import array

from fixture import H, SCHEMA, SPLITS, W

SCALE = 255.0


class PreprocessError(ValueError):
    pass


class ContaminationError(PreprocessError):
    pass


class Float32Image:
    """float32 image with explicit channel axis: shape (H, W, 1)."""

    dtype = "float32"
    shape = (H, W, 1)

    def __init__(self, flat_values):
        self._data = array("f", flat_values)
        if self._data.itemsize != 4:
            raise PreprocessError("float32 array expected")

    @property
    def flat(self):
        return list(self._data)

    def tolist(self):
        return [[[self._data[r * W + c]] for c in range(W)] for r in range(H)]


def _check_pixel(v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise PreprocessError("pixel must be an int, got %r" % (v,))
    if not 0 <= v <= 255:
        raise PreprocessError("pixel out of range 0..255: %d" % v)


def validate_image(image):
    if not isinstance(image, (list, tuple)) or len(image) != H:
        raise PreprocessError("invalid dimensions: expected %d rows" % H)
    for row in image:
        if not isinstance(row, (list, tuple)) or len(row) != W:
            raise PreprocessError("invalid dimensions: expected %dx%d rows of pixels" % (H, W))
        for v in row:
            if isinstance(v, (list, tuple)):
                raise PreprocessError("invalid dimensions: input already has a channel axis")
            _check_pixel(v)


class Scaler:
    PUBLIC = ("fit", "transform", "transform_batch", "state")

    def __init__(self):
        self._train_ids = None

    def fit(self, records):
        """Fit on TRAIN records only; anything else is contamination."""
        ids = []
        for rec in records:
            missing = [f for f in SCHEMA if f not in rec]
            if missing:
                raise PreprocessError("missing field(s): %s" % ", ".join(missing))
            split = SPLITS.get(rec["group_id"])
            if split != "train":
                raise ContaminationError(
                    "record %s (group %s, split %r) is not training data"
                    % (rec["record_id"], rec["group_id"], split)
                )
            ids.append(rec["record_id"])
        if not ids:
            raise PreprocessError("fit needs at least one training record")
        if len(set(ids)) != len(ids):
            raise PreprocessError("duplicate record_id in fit input")
        self._train_ids = frozenset(ids)  # assigned only after every check passed

    def state(self):
        return (self._train_ids, SCALE)

    def transform(self, image):
        if self._train_ids is None:
            raise PreprocessError("scaler is not fitted")
        validate_image(image)
        return Float32Image([v / SCALE for row in image for v in row])

    def transform_batch(self, images):
        return [self.transform(img) for img in images]


def serving_request(images, signature="serving_default"):
    """Body for POST /v1/models/<name>:predict (built offline, never sent)."""
    return json.dumps({
        "signature_name": signature,
        "instances": [img.tolist() for img in images],
    })
