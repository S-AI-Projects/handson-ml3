"""Deterministic split manifest for an unsupervised fixture.

- Records are grouped into "units". Records that share a `group` or a `user`
  value (directly or through a chain) always land in the same unit, so group
  and user isolation is preserved where those fields are present.
- If every record has a `time` field, units are split chronologically
  (train earliest, then validation, then test). Otherwise units are ordered by
  a SHA-256 hash of (seed, unit key), which is stable across runs and machines.
- The manifest lists the exact record IDs of each split.
"""
import hashlib
import json

SPLIT_NAMES = ("train", "validation", "test")
ISOLATION_FIELDS = ("group", "user")
HOLDOUT_FRACTION = 0.2


class ManifestError(Exception):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _check_records(records):
    problems, seen = [], set()
    for index, rec in enumerate(records):
        rid = rec.get("record_id") if isinstance(rec, dict) else None
        if not isinstance(rid, str) or not rid.strip():
            problems.append(f"record at index {index}: missing or invalid record_id")
        elif rid in seen:
            problems.append(f"duplicate record_id {rid}")
        else:
            seen.add(rid)
    if problems:
        raise ManifestError(problems)


def _units(records):
    parent = {r["record_id"]: r["record_id"] for r in records}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for field in ISOLATION_FIELDS:
        first_seen = {}
        for rec in records:
            if field in rec:
                value = rec[field]
                if value in first_seen:
                    union(first_seen[value], rec["record_id"])
                else:
                    first_seen[value] = rec["record_id"]

    units = {}
    for rec in records:
        units.setdefault(find(rec["record_id"]), []).append(rec)
    return list(units.values())


def make_manifest(records, seed=0):
    _check_records(records)
    flags = ["time" in r for r in records]
    if any(flags) and not all(flags):
        raise ManifestError(["time is present on some records but not all"])

    units = _units(records)
    if records and all(flags):
        strategy = "chronological"
        units.sort(
            key=lambda u: (
                min(r["time"] for r in u),
                max(r["time"] for r in u),
                min(r["record_id"] for r in u),
            )
        )
    else:
        strategy = "seeded-hash"
        units.sort(
            key=lambda u: hashlib.sha256(
                f"{seed}:{min(r['record_id'] for r in u)}".encode()
            ).hexdigest()
        )

    n_hold = max(1, round(len(units) * HOLDOUT_FRACTION))
    n_train = len(units) - 2 * n_hold
    if n_train < 1:
        raise ManifestError(
            [f"need at least 3 units to make three splits, got {len(units)}"]
        )

    chosen = {
        "train": units[:n_train],
        "validation": units[n_train : n_train + n_hold],
        "test": units[n_train + n_hold :],
    }
    manifest = {
        "seed": seed,
        "strategy": strategy,
        "splits": {
            name: sorted(r["record_id"] for u in chosen[name] for r in u)
            for name in SPLIT_NAMES
        },
    }
    validate_manifest(manifest, records)
    return manifest


def validate_manifest(manifest, records):
    """Raise ManifestError listing every problem found."""
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ManifestError([f"splits must be exactly {list(SPLIT_NAMES)}"])

    problems = []
    by_id = {r["record_id"]: r for r in records}
    owner = {}
    for name in SPLIT_NAMES:
        ids = splits[name]
        if not ids:
            problems.append(f"split {name} is empty")
        if len(ids) != len(set(ids)):
            problems.append(f"split {name} lists a record_id more than once")
        for rid in set(ids):
            owner.setdefault(rid, []).append(name)

    for rid in sorted(owner):
        if len(owner[rid]) > 1:
            problems.append(f"record_id {rid} appears in both {' and '.join(owner[rid])}")
        if rid not in by_id:
            problems.append(f"record_id {rid} is not in the input records")
    for rid in sorted(by_id):
        if rid not in owner:
            problems.append(f"record_id {rid} is missing from every split")

    for field in ISOLATION_FIELDS:
        seen = {}
        for rid in sorted(owner):
            rec = by_id.get(rid)
            if rec is not None and field in rec:
                seen.setdefault(rec[field], set()).update(owner[rid])
        for value in sorted(seen, key=str):
            if len(seen[value]) > 1:
                problems.append(
                    f"{field} {value} is in more than one split: {sorted(seen[value])}"
                )

    if records and all("time" in r for r in records):
        spans = {}
        for name in SPLIT_NAMES:
            times = [by_id[i]["time"] for i in splits[name] if i in by_id]
            spans[name] = (min(times), max(times)) if times else None
        for earlier, later in (("train", "validation"), ("validation", "test")):
            if spans[earlier] and spans[later] and spans[earlier][1] > spans[later][0]:
                problems.append(
                    f"{earlier} is not before {later}: latest {earlier} time "
                    f"{spans[earlier][1]} is after earliest {later} time {spans[later][0]}"
                )

    if problems:
        raise ManifestError(problems)


def write_manifest(manifest, path):
    with open(path, "w") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    from pathlib import Path

    import fixture

    out = Path(__file__).resolve().parent / "split_manifest.json"
    write_manifest(make_manifest(fixture.RECORDS, seed=fixture.SEED), out)
    print(f"wrote {out}")
