# R33-T012: Leakage-Free Classification Splitting

This lab demonstrates deterministic train, validation, and test splitting
using a small synthetic classification dataset.

The fixture is simulated data for pipeline verification only.
It does not represent real-world or clinical performance.

## Data contract

Each record contains:
- record_id
- group_id
- x1
- x2
- label

## Split design

The dataset contains eight synthetic records with labels:
- r1, r2, r3, r4: class 0
- r5, r6, r7, r8: class 1

The fixed split is:
- Train: r1, r2, r5, r6
- Validation: r3, r7
- Test: r4, r8

All splits must have disjoint record IDs and group IDs.

## Run tests

From the repository root:

python3 -m unittest discover -s course-labs/r33-t012 -p "test_*.py"
