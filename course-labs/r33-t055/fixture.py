"""SYNTHETIC 8-row fixture. NOT observations from the named real dataset.
Pipeline verification only; not evidence of real-world or clinical performance."""

SCHEMA = ("record_id", "group_id", "x1", "x2", "label")

_X1 = [0, 1, 2, 3, 4, 5, 6, 7]
_LABEL = [0, 0, 0, 0, 1, 1, 1, 1]
# Each group holds one row of each class: g1=(r0,r4) g2=(r1,r5) g3=(r2,r6) g4=(r3,r7)
_GROUP = ["g1", "g2", "g3", "g4", "g1", "g2", "g3", "g4"]

RECORDS = [
    {"record_id": f"r{i}", "group_id": _GROUP[i], "x1": _X1[i], "x2": 1, "label": _LABEL[i]}
    for i in range(8)
]

# Fixed, group-disjoint, stratified split (every split has equal class counts).
SPLITS = {"train": ["g1", "g2"], "val": ["g3"], "test": ["g4"]}


def get_split(name):
    groups = SPLITS[name]
    return [r for r in RECORDS if r["group_id"] in groups]
