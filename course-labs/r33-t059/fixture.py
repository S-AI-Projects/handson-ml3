"""Literal synthetic CPU smoke fixture (NOT the named real dataset).

Schema: record_id (str), group_id (str), pixels (16 ints, row-major 4x4,
0..255), label (int).  Three all-zero images labelled 0 and three images with
a white (255) main diagonal labelled 1.  Each record has a unique record_id;
group_ids are declared explicitly (g0..g2, two records each, one per class).
"""

H = W = 4
SCHEMA = ("record_id", "group_id", "pixels", "label")
KNOWN_LABELS = (0, 1)
FALLBACK_LABEL = -1  # explicit fallback for an unseen category


def _diag():
    return [255 if r == c else 0 for r in range(H) for c in range(W)]


FIXTURE = [
    {"record_id": "r0", "group_id": "g0", "pixels": [0] * 16, "label": 0},
    {"record_id": "r1", "group_id": "g0", "pixels": _diag(), "label": 1},
    {"record_id": "r2", "group_id": "g1", "pixels": [0] * 16, "label": 0},
    {"record_id": "r3", "group_id": "g1", "pixels": _diag(), "label": 1},
    {"record_id": "r4", "group_id": "g2", "pixels": [0] * 16, "label": 0},
    {"record_id": "r5", "group_id": "g2", "pixels": _diag(), "label": 1},
]
