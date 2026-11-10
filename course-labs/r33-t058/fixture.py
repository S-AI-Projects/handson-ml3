"""SYNTHETIC regression fixture: a deterministic formula, target = 2*x1 + 1.
NOT observations from the named real dataset. Pipeline verification only;
not evidence of real-world or clinical performance."""

SCHEMA = ("record_id", "x1", "x2", "target")

RECORDS = [
    {"record_id": f"r{i + 1}", "x1": i, "x2": 1, "target": 2 * i + 1}
    for i in range(6)
]

# Explicit train / validation / test separation (by record ID)
SPLITS = {"train": ["r1", "r2", "r3", "r4"], "val": ["r5"], "test": ["r6"]}


def get_split(name):
    ids = SPLITS[name]
    return [r for r in RECORDS if r["record_id"] in ids]
