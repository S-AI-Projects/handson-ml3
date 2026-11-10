# R33-T060 - Make preprocessing explicit: Querying TF Serving through the REST API (19 training and deploying at scale)

Bounded component for the image interface, not a full reproduction of the original notebook
(19_training_and_deploying_at_scale.ipynb). No network call is made; the TF Serving REST request
body is built offline only.

## Data contract
Input: 4x4 nested list of ints, each 0..255 (rows of pixels, no channel axis).
Output: float32 image, shape (4, 4, 1) with an explicit channel axis, values pixel / 255.0 (range 0.0 .. 1.0).

Rejected inputs:
- wrong dimensions: not 4 rows of 4, ragged, flat list, or already 3-D with a channel axis
- out-of-range pixels: below 0 or above 255
- non-integer pixels: floats, booleans, strings, None

## Fixture (synthetic CPU smoke fixture, not the named real dataset)
Six 4x4 images: three all-zero labelled 0 and three with a white (255) main diagonal labelled 1.
Unique record IDs r0..r5 and group IDs g0..g2. Splits are declared per group, so no record is in
two splits: g0 = train (r0, r1), g1 = validation (r2, r3), g2 = test (r4, r5).

## Leakage control
- The only fitting entry point is Scaler.fit(), and it accepts training-split records only.
- Any validation or test record passed to fit() raises ContaminationError and leaves the state unchanged.
- transform() is pure: it never changes fitted state, so held-out input cannot be fitted through the
  public interface. There is no fit_transform.
- The 0..255 -> 0..1 scaling is a fixed-range transform with no learned statistics, so fit() only
  records which training records the scaler was fitted on. It is a contamination gate, and this is stated
  honestly rather than claiming data-dependent fitting.

## Run
    python3 -m unittest discover -s course-labs/r33-t060 -p "test_*.py"
    python3 course-labs/r33-t060/check.py

check.py requires a nonzero test count, writes inputs, expected versus actual results and tool versions
to work/R33-T060/, exits non-zero on failure, and also runs the suite with a deliberately changed
expected value, which must fail.

## Limits
- Synthetic fixture results are pipeline verification only, not evidence of real-world or clinical performance.
- Float32 values are checked against the 0..1 contract; no claim about a real TF Serving model is made.
- Optional real-data extension (not done) would need its own recorded source, licence, access
  requirements and train/validation/test split design.
- Remove work/R33-T060/ from a later upstream branch.
