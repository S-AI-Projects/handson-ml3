"""
R33-T035: Make preprocessing explicit
======================================
Project/section: 09 unsupervised learning / Multiple Initializations
Notebook: 09_unsupervised_learning.ipynb

Goal
----
Fit any scaling or projection on training data only and retain the fitted
parameters so the same transform can be applied to held-out data without
leaking information from those rows.

Implements a MinMaxScaler that fits exclusively on training rows and a
nearest-centroid classifier with deterministic (fixed-seed) centroid selection.

Fixture
-------
Six 2-D synthetic points (simulated, NOT from any real dataset):
  r1=(0,0), r2=(0,1), r3=(1,0), r4=(5,5), r5=(5,6), r6=(6,5)
Fixed centroids: c0=(0,0), c1=(5,5)
Expected nearest-centroid assignments: [0, 0, 0, 1, 1, 1]

Data contract
-------------
* Source: Fully synthetic, generated for pipeline verification only.
* Licence: N/A
* Real dataset used: No
* Disclaimer: Results are pipeline verification only and must NOT be cited
  as evidence of real-world or clinical performance.
"""

import math


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

FIXTURE = [
    {"record_id": "r1", "x1": 0.0, "x2": 0.0},
    {"record_id": "r2", "x1": 0.0, "x2": 1.0},
    {"record_id": "r3", "x1": 1.0, "x2": 0.0},
    {"record_id": "r4", "x1": 5.0, "x2": 5.0},
    {"record_id": "r5", "x1": 5.0, "x2": 6.0},
    {"record_id": "r6", "x1": 6.0, "x2": 5.0},
]

FIXED_CENTROIDS = [(0.0, 0.0), (5.0, 5.0)]

EXPECTED_ASSIGNMENTS = [0, 0, 0, 1, 1, 1]  # one per fixture row in order


# ---------------------------------------------------------------------------
# MinMaxScaler — fit only on training data, retain parameters for transform
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """
    Standard min-max feature scaler.

    Usage:
        scaler = MinMaxScaler()
        scaler.fit(train_rows)        # learns min/max from train only
        scaled_train = scaler.transform(train_rows)
        scaled_test  = scaler.transform(test_rows)   # uses same params

    Raises
    ------
    RuntimeError  if transform() is called before fit().
    ValueError    if a column has zero range (constant feature).
    """

    def __init__(self):
        self._fitted = False
        self._min = None
        self._max = None
        self._range = None

    # ------------------------------------------------------------------
    def fit(self, rows: list[dict]) -> "MinMaxScaler":
        """Learn min and max of x1, x2 from *rows* only."""
        if not rows:
            raise ValueError("fit() requires at least one row.")
        x1_vals = [r["x1"] for r in rows]
        x2_vals = [r["x2"] for r in rows]
        self._min = (min(x1_vals), min(x2_vals))
        self._max = (max(x1_vals), max(x2_vals))
        range_x1 = self._max[0] - self._min[0]
        range_x2 = self._max[1] - self._min[1]
        if range_x1 == 0.0:
            raise ValueError("x1 is constant in training data — cannot scale.")
        if range_x2 == 0.0:
            raise ValueError("x2 is constant in training data — cannot scale.")
        self._range = (range_x1, range_x2)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, rows: list[dict]) -> list[tuple[float, float]]:
        """
        Apply the fitted scaling to *rows*.

        Returns a list of (scaled_x1, scaled_x2) tuples.

        Raises RuntimeError if called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "MinMaxScaler.transform() called before fit(). "
                "Call fit(train_rows) first."
            )
        result = []
        for r in rows:
            sx1 = (r["x1"] - self._min[0]) / self._range[0]
            sx2 = (r["x2"] - self._min[1]) / self._range[1]
            result.append((sx1, sx2))
        return result

    # ------------------------------------------------------------------
    def fit_transform(self, rows: list[dict]) -> list[tuple[float, float]]:
        """Convenience: fit then transform the same rows."""
        return self.fit(rows).transform(rows)

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------

def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Nearest-centroid assignment
# ---------------------------------------------------------------------------

def nearest_centroid_assignments(
    points: list[tuple[float, float]],
    centroids: list[tuple[float, float]],
) -> list[int]:
    """
    Assign each point in *points* to the index of the nearest centroid.

    Parameters
    ----------
    points    : list of (x1, x2) tuples
    centroids : list of (cx1, cx2) centroids

    Returns
    -------
    List of integer cluster indices, one per point.

    Raises
    ------
    ValueError if centroids list is empty.
    """
    if not centroids:
        raise ValueError("centroids list must not be empty.")
    assignments = []
    for pt in points:
        dists = [euclidean(pt, c) for c in centroids]
        assignments.append(dists.index(min(dists)))
    return assignments


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    train_rows: list[dict],
    test_rows: list[dict],
    centroids: list[tuple[float, float]],
) -> dict:
    """
    Full preprocessing + nearest-centroid pipeline.

    1. Fit MinMaxScaler on *train_rows* only.
    2. Scale both train and test rows with the fitted parameters.
    3. Scale the centroids using the same fitted parameters.
    4. Assign each test point to the nearest (scaled) centroid.

    Returns a dict with keys:
        scaler          : fitted MinMaxScaler instance
        scaled_train    : scaled training coordinates
        scaled_test     : scaled test coordinates
        scaled_centroids: centroids in scaled space
        assignments     : cluster index per test row
        record_ids      : record_id list from test_rows
    """
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_rows)

    # Reuse fitted scaler — no re-fitting on test data
    scaled_test = scaler.transform(test_rows)

    # Scale centroids using the same fitted parameters
    centroid_rows = [{"x1": c[0], "x2": c[1]} for c in centroids]
    scaled_centroids = scaler.transform(centroid_rows)

    assignments = nearest_centroid_assignments(scaled_test, scaled_centroids)
    record_ids = [r["record_id"] for r in test_rows]

    return {
        "scaler": scaler,
        "scaled_train": scaled_train,
        "scaled_test": scaled_test,
        "scaled_centroids": scaled_centroids,
        "assignments": assignments,
        "record_ids": record_ids,
    }


# ---------------------------------------------------------------------------
# Demo (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = FIXTURE
    result = run_pipeline(rows, rows, FIXED_CENTROIDS)
    print("record_id  assignment")
    for rid, asgn in zip(result["record_ids"], result["assignments"]):
        print(f"  {rid}        {asgn}")
    print(f"\nExpected: {EXPECTED_ASSIGNMENTS}")
    print(f"Got:      {result['assignments']}")
    assert result["assignments"] == EXPECTED_ASSIGNMENTS, "Mismatch!"
    print("\nAll assertions passed.")
