# R33-T013: Minimal Image Classification Baseline

## Purpose

This scoped contribution implements a majority-class baseline for a
small synthetic image-classification fixture associated with the
"Regression MLPs" section of the neural networks chapter.

It is a CPU-only pipeline smoke test. It does not reproduce the full
chapter or train a neural network.

## Data contract

Each record contains:

- `record_id`: unique non-empty string
- `group_id`: non-empty group identifier
- `pixels`: 4x4 grayscale matrix of integer values from 0 to 255
- `label`: integer class label, 0 or 1

The fixture contains six synthetic images:

- Three all-zero images labelled 0
- Three images with a white diagonal labelled 1

The records are deterministic formula data created for this task.
They are not the chapter's real dataset.

## Fixed splits

- Training: r1, r2, r4, r5
- Validation: r6
- Test: r3

Record IDs and group IDs are checked for split leakage.

## Baseline

The baseline learns the most frequent label from the training records
only. If training labels are tied, the smaller label is selected.

The selected label is predicted for each test record. Accuracy is
calculated only on the fixed test split.

For this fixture, the training labels are tied, so the deterministic
tie-break selects label 0. The test record r3 has label 0, giving
a smoke-fixture test accuracy of 1.0.

This one-record result is only pipeline verification. It is not a
real-world accuracy claim or evidence of clinical performance.

## Run tests

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r33-t013 -p "test_*.py" -v


## Step 7. Link the task from the root README

This adds a link to your new task documentation without replacing existing content.

```bash
cat >> README.md <<'EOF'

### Student task contribution: R33-T013

See [R33-T013 minimal image classification baseline](course-labs/r33-t013/README.md).
