"""Nearest-centroid assignment (the assignment step of K-Means), stdlib only."""


def nearest_centroid(records, centroids):
    """Return the index of the nearest centroid for each record.

    Ties go to the lowest centroid index, so the result is deterministic.
    """
    assignments = []
    for rec in records:
        distances = [
            (rec["x1"] - cx) ** 2 + (rec["x2"] - cy) ** 2 for cx, cy in centroids
        ]
        assignments.append(distances.index(min(distances)))
    return assignments
