"""SYNTHETIC 4x4 grayscale smoke fixture. NOT the named real dataset.
Pipeline verification only; not evidence of real-world or clinical performance.
Group IDs are declared per record; splits are defined by the saved config."""

SCHEMA = ("record_id", "group_id", "pixels", "label")


def _blank():
    return [[0] * 4 for _ in range(4)]


def _diagonal():
    img = _blank()
    for i in range(4):
        img[i][i] = 255
    return img


RECORDS = [
    {"record_id": "r0", "group_id": "g1", "pixels": _blank(),    "label": 0},
    {"record_id": "r1", "group_id": "g1", "pixels": _diagonal(), "label": 1},
    {"record_id": "r2", "group_id": "g2", "pixels": _blank(),    "label": 0},
    {"record_id": "r3", "group_id": "g2", "pixels": _diagonal(), "label": 1},
    {"record_id": "r4", "group_id": "g3", "pixels": _blank(),    "label": 0},
    {"record_id": "r5", "group_id": "g3", "pixels": _diagonal(), "label": 1},
]
