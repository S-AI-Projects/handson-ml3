# R33-T059: order-invariant compressed TFRecord pipeline

Refs #59. Stdlib-only CPU smoke lab.

## Property tested
Changing the record order of a compressed TFRecord file does not change the
documented result once records are aligned by `record_id` (seed 59). Time
order is not applicable to the image fixture and is not randomised.

## Data contract
Each record: `record_id` (unique str), `group_id` (str), `pixels` (4x4 ints
0..255), `label` (int). Files are gzip-compressed TFRecord framing
(length, masked CRC32C, payload, masked CRC32C) with JSON payloads.

## Unsupported inputs
- Missing field -> `ValueError("missing field: ...")`
- Shape other than 4x4 -> `ValueError("shape mismatch ...")`
- Unseen label -> mapped to explicit fallback `-1`, excluded from accuracy

## Fixture
Six synthetic 4x4 grayscale images: three all-zero (label 0), three with a
white diagonal of 255 (label 1). Unique record IDs, group IDs `g0..g2`.
CPU smoke fixture, not the named real dataset. A real-data extension would
need its own recorded source, licence, access requirements and split design.

## Run

    python3 -m unittest discover -s course-labs/r33-t059 -p "test_*.py"
    python3 course-labs/r33-t059/check.py   # exits 1 by design (changed expectation)

Evidence is written to `work/R33-T059/` (remove it from a later upstream branch).

## Limits
Results are pipeline verification only, not evidence of real-world or
clinical performance. No broad adversarial-robustness claim. The stdlib
TFRecord codec has not been cross-read by TensorFlow.
