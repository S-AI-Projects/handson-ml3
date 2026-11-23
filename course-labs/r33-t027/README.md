# Course Lab R33-T027: Reproducible Configuration

## Overview
Implements a serializable configuration contract recording seed, schema version, feature ordering, split IDs, and algorithm settings with strict validation against unknown and contradictory parameters.

## Configuration Schema
- `seed`: integer non-negative random seed.
- `schema_version`: semantic versioning tag (`"1.0"`).
- `feature_ordering`: ordered list of expected input features (`["x1", "x2"]`).
- `split_ids`: disjoint train, validation, and test record ID sets.
- `algorithm_settings`: serializable model hyperparameters.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t027 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
