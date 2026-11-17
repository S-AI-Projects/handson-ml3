"""
Course Lab R33-T002: Deterministic Split Manifest & Leakage-Free Splitting.

Scope: 04 training linear models / The Normal Equation regression fixture.
Note: Verification fixture rows are synthetic and follow target = 2*x1 + 1.
They are strictly intended for split leakage prevention and pipeline verification.
They are NOT observations from any real dataset and must not be used for performance claims.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import csv
import json
import math
from pathlib import Path


class LeakageError(Exception):
    """Raised when record, group, or temporal leakage is detected across splits."""
    pass


class SchemaValidationError(Exception):
    """Raised when record fields fail schema or type contract."""
    pass


@dataclass
class SplitManifest:
    """Represents a deterministic train/validation/test split manifest."""
    manifest_version: str
    split_strategy: str
    train_record_ids: List[str]
    validation_record_ids: List[str]
    test_record_ids: List[str]
    counts: Dict[str, int]
    is_disjoint: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "split_strategy": self.split_strategy,
            "train_record_ids": self.train_record_ids,
            "validation_record_ids": self.validation_record_ids,
            "test_record_ids": self.test_record_ids,
            "counts": self.counts,
            "is_disjoint": self.is_disjoint,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def validate_regression_schema(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates the R33-T002 regression schema:
      - record_id: unique non-empty string or integer.
      - x1: numeric scalar (int or float, non-NaN).
      - x2: numeric scalar (int or float, non-NaN).
      - target: numeric scalar (int or float, non-NaN).
    """
    if not records:
        raise SchemaValidationError("Dataset cannot be empty.")

    seen_ids: Set[str] = set()
    cleaned_records: List[Dict[str, Any]] = []

    for idx, r in enumerate(records):
        rec_id = r.get("record_id")
        if rec_id is None or str(rec_id).strip() == "":
            raise SchemaValidationError(f"Row {idx}: missing or empty 'record_id'")
        
        str_id = str(rec_id).strip()
        if str_id in seen_ids:
            raise SchemaValidationError(f"Duplicate record_id detected: {str_id!r}")
        seen_ids.add(str_id)

        for field in ("x1", "x2", "target"):
            if field not in r or r[field] is None:
                raise SchemaValidationError(f"Row {str_id}: missing required numeric field '{field}'")
            val = r[field]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise SchemaValidationError(
                    f"Row {str_id}: field '{field}' must be int or float, got {type(val).__name__} ({val!r})"
                )
            if math.isnan(val) or math.isinf(val):
                raise SchemaValidationError(
                    f"Row {str_id}: field '{field}' must be finite and non-NaN, got {val!r}"
                )

        cleaned_records.append(r)

    return cleaned_records


def build_deterministic_split_manifest(
    records: List[Dict[str, Any]],
    train_ids: Optional[List[str]] = None,
    validation_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
    strategy: str = "deterministic_formula",
) -> SplitManifest:
    """
    Builds and strictly verifies a deterministic split manifest for regression records.
    Default IDs conform to R33-T002 fixture:
      - Train: r1..r4
      - Validation: r5
      - Test: r6
    """
    validate_regression_schema(records)
    all_record_ids = {str(r["record_id"]).strip() for r in records}
    record_map = {str(r["record_id"]).strip(): r for r in records}

    if train_ids is None:
        train_ids = ["r1", "r2", "r3", "r4"]
    if validation_ids is None:
        validation_ids = ["r5"]
    if test_ids is None:
        test_ids = ["r6"]

    train_set = set(train_ids)
    val_set = set(validation_ids)
    test_set = set(test_ids)

    # 1. Existence check
    for split_name, s_set in [("train", train_set), ("validation", val_set), ("test", test_set)]:
        missing = s_set - all_record_ids
        if missing:
            raise KeyError(f"Specified {split_name} record IDs not found in dataset: {sorted(missing)}")

    # 2. Strict disjoint check (No record leakage)
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap or train_test_overlap or val_test_overlap:
        overlaps = {
            "train_val": sorted(train_val_overlap),
            "train_test": sorted(train_test_overlap),
            "val_test": sorted(val_test_overlap),
        }
        raise LeakageError(f"Split leakage detected: record sets are not disjoint. Overlaps: {overlaps}")

    # 3. Group isolation check (No group leakage where group_id is present)
    groups_train = {record_map[i]["group_id"] for i in train_ids if "group_id" in record_map[i] and record_map[i]["group_id"]}
    groups_val = {record_map[i]["group_id"] for i in validation_ids if "group_id" in record_map[i] and record_map[i]["group_id"]}
    groups_test = {record_map[i]["group_id"] for i in test_ids if "group_id" in record_map[i] and record_map[i]["group_id"]}

    grp_overlap_tv = groups_train & groups_val
    grp_overlap_tt = groups_train & groups_test
    grp_overlap_vt = groups_val & groups_test

    if grp_overlap_tv or grp_overlap_tt or grp_overlap_vt:
        grp_overlaps = {
            "train_val_group_overlap": sorted(grp_overlap_tv),
            "train_test_group_overlap": sorted(grp_overlap_tt),
            "val_test_group_overlap": sorted(grp_overlap_vt),
        }
        raise LeakageError(f"Group leakage detected across splits: {grp_overlaps}")

    # 4. Chronological order check where timestamps are present
    has_timestamps = all("timestamp" in r for r in records)
    if has_timestamps:
        max_train_ts = max(record_map[i]["timestamp"] for i in train_ids)
        min_val_ts = min(record_map[i]["timestamp"] for i in validation_ids)
        max_val_ts = max(record_map[i]["timestamp"] for i in validation_ids)
        min_test_ts = min(record_map[i]["timestamp"] for i in test_ids)

        if max_train_ts >= min_val_ts:
            raise LeakageError(
                f"Temporal leakage: train max timestamp ({max_train_ts}) >= validation min timestamp ({min_val_ts})"
            )
        if max_val_ts >= min_test_ts:
            raise LeakageError(
                f"Temporal leakage: validation max timestamp ({max_val_ts}) >= test min timestamp ({min_test_ts})"
            )

    manifest = SplitManifest(
        manifest_version="1.0",
        split_strategy=strategy,
        train_record_ids=list(train_ids),
        validation_record_ids=list(validation_ids),
        test_record_ids=list(test_ids),
        counts={
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
            "total": len(train_ids) + len(validation_ids) + len(test_ids),
        },
        is_disjoint=True,
    )
    return manifest


def load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Loads records from a JSON or CSV file, or returns existing list of dicts."""
    if isinstance(source, list):
        return source

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON must be a top-level list of record objects.")
            return data

    if p.suffix.lower() == ".csv":
        records = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row: Dict[str, Any] = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    v_clean = v.strip()
                    if k_clean in ("x1", "x2", "target", "timestamp"):
                        try:
                            parsed_row[k_clean] = float(v_clean) if "." in v_clean else int(v_clean)
                        except ValueError:
                            parsed_row[k_clean] = v_clean
                    else:
                        parsed_row[k_clean] = v_clean
                records.append(parsed_row)
        return records

    raise ValueError(f"Unsupported format {p.suffix}")
