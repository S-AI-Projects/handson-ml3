"""SYNTHETIC pipeline fixture (not a real corpus, not Shakespeare data)."""

SEED = 0  # fixed seed; nothing here is random, kept for explicitness

TEXTS = ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"]
LABELS = [1, 1, 0, 0, 1, 0]
IDS = [f"t{i}" for i in range(1, 7)]

RECORDS = [
    {"record_id": i, "text": t, "label": l}
    for i, t, l in zip(IDS, TEXTS, LABELS)
]

# Explicit, disjoint split
SPLITS = {
    "train": ["t1", "t2", "t3", "t4"],
    "validation": ["t5"],
    "test": ["t6"],
}
