# Course Lab R33-T042: Deterministic Split Manifest & Centroid Assignment

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T042 (Task 2 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T042`  
**Target Item:** `08_dimensionality_reduction.ipynb` (`## Principal Components`)

---

## 1. Scope & Objective
Implements a deterministic, leakage-preventing data split manifest and verifiable nearest-centroid assignment for unsupervised 2D point representations.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The 6-point 2D fixture in `fixture.json` `((0,0), (0,1), (1,0), (5,5), (5,6), (6,5))` is a synthetic smoke fixture intended exclusively for offline geometric verification. It does not represent high-dimensional PCA datasets.

---

## 2. Nearest Centroid Invariants
Given fixed reference centroids $c_0 = (0, 0)$ and $c_1 = (5, 5)$, points are partitioned by Euclidean distance:
- Points $p_1(0,0), p_2(0,1), p_3(1,0)$ are closest to $c_0 \implies$ cluster 0
- Points $p_4(5,5), p_5(5,6), p_6(6,5)$ are closest to $c_1 \implies$ cluster 1
- **Assignments:** `0, 0, 0, 1, 1, 1`

---

## 3. Disjoint Partitioning Rules
- Record sets between train, validation, and test must have empty intersections.
- Manifests with overlapping record IDs are strictly rejected.

---

## 4. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t042 -p "test_*.py"
```
