# Course Lab R33-T030: Schema Validator

## Overview
Implements a strict schema validator for classification tabular input contracts, asserting `record_id`, `group_id`, `x1`, `x2`, and `label` without silently dropping records.

## Validation Contract
- Identifiers: Checks existence, non-emptiness, and uniqueness of `record_id`.
- Typing: Validates numeric types for features `x1` and `x2`, string types for `group_id`, and binary `0/1` for `label`.
- Integrity: Explicitly reports record ID, field, and reason for every violation without record omission.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t030 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
