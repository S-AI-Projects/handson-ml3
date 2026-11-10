"""SYNTHETIC pipeline fixture for K-Means (not a real dataset).

Six fictional 2-D points with stable IDs. Results computed from this fixture
are pipeline verification only, not evidence of real-world performance.
"""

SEED = 0

IDS = ["p1", "p2", "p3", "p4", "p5", "p6"]
POINTS = [(0, 0), (0, 1), (1, 0), (5, 5), (5, 6), (6, 5)]

# Schema: record_id, x1, x2
RECORDS = [
    {"record_id": i, "x1": x1, "x2": x2}
    for i, (x1, x2) in zip(IDS, POINTS)
]

CENTROIDS = [(0, 0), (5, 5)]
EXPECTED_ASSIGNMENTS = [0, 0, 0, 1, 1, 1]
