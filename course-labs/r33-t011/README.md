# R33-T011: Regression Data Contract Validation

## Overview

This lab demonstrates how to validate a small regression dataset before
passing it into a machine learning pipeline.

It is an additional synthetic exercise associated with Chapter 2,
"End-to-End Machine Learning Project".

## Data contract

Every record must contain exactly these fields:

| Field | Requirement |
|---|---|
| `record_id` | Non-empty, unique string |
| `x1` | Finite numeric scalar |
| `x2` | Finite numeric scalar |
| `target` | Finite numeric scalar |

The validator rejects missing fields, duplicate IDs, text in numeric
fields, Boolean values, non-finite numbers, nested values, malformed
records, and unexpected fields.

Validation errors report the record ID, field, and reason.

Invalid records are not silently dropped.

## Synthetic fixture

The fixture is deliberately simulated and contains six records.

Its deterministic formula is:

`target = 2 * x1 + 1`

The required split is:

- Training: r1, r2, r3, r4
- Validation: r5
- Test: r6

The fixture includes seed metadata (`42`) for reproducibility.
No external dataset or network access is required.

## Running the tests

From the repository root, run:

```bash
python3 -m unittest discover -s course-labs/r33-t011 -p "test_*.py" -v
```

The tests use Python's standard library and run on CPU without downloading
data or installing machine learning packages.

## Files

- `pipeline.py`: schema and split validation functions.
- `test_pipeline.py`: automated valid and invalid input tests.
- `fixture.json`: deterministic synthetic dataset and split definitions.

## Limitations

This synthetic fixture is intended only for pipeline and data-contract
verification. It does not represent the California housing dataset used
in the book and must not be interpreted as real-world predictive,
statistical, or clinical performance.

The fixed six-record split is a verification example, not a general
recommendation for splitting production datasets.
