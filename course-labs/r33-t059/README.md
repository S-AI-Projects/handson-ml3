# R33-T059 - Input perturbation test: Compressed TFRecord Files (13 loading and preprocessing data)

Bounded component for the image interface, not a full reproduction of the original notebook
(13_loading_and_preprocessing_data.ipynb).

## What was tested
Property: changing the record order of a compressed TFRecord file does not change the documented
result once records are aligned by their stable record_id.
Time order is not applicable to this image modality and is never randomised.

## Data contract (schema)
- record_id: string, unique per record
- group_id: string, declared group (g0..g2)
- pixels: 16 ints, row-major 4x4 grayscale, each 0..255
- label: int, known labels 0 and 1

File format: TFRecord, GZIP-compressed, tf.train.Example records, masked CRC32C framing.
Written and read by a stdlib-only codec (tfrecord_io.py) because TensorFlow was not available;
it follows the TFRecord spec but has not been cross-read by TensorFlow.

## Fixture (synthetic CPU smoke fixture, not the named real dataset)
Six 4x4 images: three all-zero labelled 0 and three with a white (255) main diagonal labelled 1;
unique record IDs r0..r5, group IDs g0..g2 (one image per class in each group).

## Documented result
Rule: predict 1 if any pixel > 0, else 0. Expected: r0, r2, r4 -> 0 and r1, r3, r5 -> 1,
accuracy 1.0, class counts {0: 3, 1: 3}.

## Unsupported inputs
- missing field: rejected (SchemaError)
- shape mismatch (e.g. 3x3): rejected (SchemaError)
- unseen label: mapped to explicit fallback -1, reported in fallback_ids, excluded from accuracy
- duplicate record_id: rejected
- corrupt CRC: rejected (ValueError)

## Run
    python3 -m unittest discover -s course-labs/r33-t059 -p "test_*.py"
    python3 course-labs/r33-t059/check.py

check.py requires a nonzero test count, writes evidence to work/R33-T059/, exits non-zero on failure,
and also runs the suite with a deliberately changed expected result, which must fail.

## Limits
- Does not claim broad adversarial robustness: one order perturbation (seed 59) plus rejected
  inputs and one fallback.
- Synthetic fixture results are pipeline verification only, not evidence of real-world or clinical performance.
- Optional real-data extension (not done) would need its own recorded source, licence, access
  requirements and train/validation/test split design. The fixture has no split; it is test-only.
- Remove work/R33-T059/ from a later upstream branch.
