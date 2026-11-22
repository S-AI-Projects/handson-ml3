"""
Course Lab R33-T042: Deterministic Split Manifest & Centroid Assignment
Implements deterministic data splitting and nearest-centroid assignment for unsupervised 2D point fixtures.
"""
import math
from typing import List, Dict, Any, Tuple, Optional

CENTROIDS = ((0.0, 0.0), (5.0, 5.0))

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def assign_nearest_centroids(records: List[Dict[str, Any]], centroids=CENTROIDS) -> List[int]:
    """
    Computes nearest-centroid assignment (0 or 1) for each 2D point.
    For fixed centroids (0,0) and (5,5), assigns points to nearest centroid.
    """
    assignments = []
    for r in records:
        pt = (float(r["x"]), float(r["y"]))
        dist0 = euclidean_distance(pt, centroids[0])
        dist1 = euclidean_distance(pt, centroids[1])
        assignments.append(0 if dist0 <= dist1 else 1)
    return assignments

def generate_split_manifest(records: List[Dict[str, Any]], 
                            train_ids: Optional[List[str]] = None,
                            val_ids: Optional[List[str]] = None,
                            test_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generates deterministic train/val/test split manifest with exact record IDs.
    Default partition: train=[p1, p2, p4, p5], val=[p3], test=[p6].
    """
    if train_ids is None:
        train_ids = ["p1", "p2", "p4", "p5"]
    if val_ids is None:
        val_ids = ["p3"]
    if test_ids is None:
        test_ids = ["p6"]

    manifest = {
        "manifest_version": "1.0.0",
        "splits": {
            "train": sorted(train_ids),
            "validation": sorted(val_ids),
            "test": sorted(test_ids)
        },
        "split_counts": {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids)
        }
    }
    validate_manifest(manifest, records)
    return manifest

def validate_manifest(manifest: Dict[str, Any], records: Optional[List[Dict[str, Any]]] = None, strict_group_isolation: bool = False) -> bool:
    """
    Validates manifest disjointness.
    Rejects any duplicate/overlapping record IDs across train, validation, and test.
    """
    splits = manifest.get("splits", {})
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("validation", []))
    test_set = set(splits.get("test", []))

    # Check for disjointness
    if train_set.intersection(val_set):
        overlap = train_set.intersection(val_set)
        raise ValueError(f"Train and validation splits overlap on record IDs: {overlap}")
    if train_set.intersection(test_set):
        overlap = train_set.intersection(test_set)
        raise ValueError(f"Train and test splits overlap on record IDs: {overlap}")
    if val_set.intersection(test_set):
        overlap = val_set.intersection(test_set)
        raise ValueError(f"Validation and test splits overlap on record IDs: {overlap}")

    if strict_group_isolation and records is not None:
        rec_to_group = {r["record_id"]: r.get("group_id") for r in records}
        train_groups = {rec_to_group.get(rid) for rid in train_set if rid in rec_to_group}
        test_groups = {rec_to_group.get(rid) for rid in test_set if rid in rec_to_group}
        if train_groups.intersection(test_groups):
            raise ValueError(f"Train and test splits overlap on groups: {train_groups.intersection(test_groups)}")

    return True
