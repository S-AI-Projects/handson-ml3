"""
Course Lab R33-T021: Deterministic Split Manifest & Leakage-Free Splitting
Pipeline verification fixture for Xavier and He Initialization (11_training_deep_neural_networks.ipynb).
Synthetic CPU fixture only - not evidence of real-world or clinical performance.
"""

import json
from typing import Dict, List, Set, Any

class DeterministicSplitManifest:
    """
    Constructs deterministic, leakage-free train/validation/test splits
    preserving group isolation and chronological ordering for time series.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def split_by_group(self, records: List[Dict[str, Any]], 
                       train_groups: List[str], 
                       val_groups: List[str], 
                       test_groups: List[str]) -> Dict[str, List[str]]:
        """
        Assigns records to splits based on explicit, disjoint group IDs.
        Enforces that groups and record IDs are strictly disjoint across splits.
        """
        # Validate group disjointness
        grp_train_set = set(train_groups)
        grp_val_set = set(val_groups)
        grp_test_set = set(test_groups)

        if not grp_train_set.isdisjoint(grp_val_set):
            raise ValueError(f"Train and validation groups overlap: {grp_train_set & grp_val_set}")
        if not grp_train_set.isdisjoint(grp_test_set):
            raise ValueError(f"Train and test groups overlap: {grp_train_set & grp_test_set}")
        if not grp_val_set.isdisjoint(grp_test_set):
            raise ValueError(f"Validation and test groups overlap: {grp_val_set & grp_test_set}")

        manifest = {"train": [], "val": [], "test": []}
        all_ids: Set[str] = set()

        for r in records:
            rid = r["record_id"]
            if rid in all_ids:
                raise ValueError(f"Duplicate record_id detected: {rid}")
            all_ids.add(rid)

            grp = r["group_id"]
            if grp in grp_train_set:
                manifest["train"].append(rid)
            elif grp in grp_val_set:
                manifest["val"].append(rid)
            elif grp in grp_test_set:
                manifest["test"].append(rid)
            else:
                raise ValueError(f"Record {rid} belongs to undeclared group: {grp}")

        # Deterministic sorting
        manifest["train"].sort()
        manifest["val"].sort()
        manifest["test"].sort()

        # Validate disjointness of record IDs
        self.validate_manifest(manifest)
        return manifest

    def split_chronological(self, records: List[Dict[str, Any]], 
                            train_ratio: float = 0.5, 
                            val_ratio: float = 0.25) -> Dict[str, List[str]]:
        """
        Deterministic chronological split based on timestamp.
        Ensures max(train_time) <= min(val_time) <= min(test_time).
        """
        sorted_records = sorted(records, key=lambda r: (r.get("timestamp", 0), r["record_id"]))
        n = len(sorted_records)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        manifest = {
            "train": [r["record_id"] for r in sorted_records[:train_end]],
            "val": [r["record_id"] for r in sorted_records[train_end:val_end]],
            "test": [r["record_id"] for r in sorted_records[val_end:]]
        }
        self.validate_manifest(manifest)
        return manifest

    def validate_manifest(self, manifest: Dict[str, List[str]]) -> None:
        """
        Asserts that train, val, and test splits are strictly disjoint.
        """
        s_train = set(manifest.get("train", []))
        s_val = set(manifest.get("val", []))
        s_test = set(manifest.get("test", []))

        if not s_train.isdisjoint(s_val):
            raise ValueError(f"Train and val splits overlap: {s_train & s_val}")
        if not s_train.isdisjoint(s_test):
            raise ValueError(f"Train and test splits overlap: {s_train & s_test}")
        if not s_val.isdisjoint(s_test):
            raise ValueError(f"Val and test splits overlap: {s_val & s_test}")
