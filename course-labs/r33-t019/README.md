# R33-T019: Image Input Robustness Testing

## Objective
Test whether changing the order of image records affects pipeline results when records have stable, unique IDs.

## Dataset
This lab uses six synthetic 4x4 grayscale images:
- Three all-zero images labelled 0.
- Three white-diagonal images labelled 1.

Each record contains record_id, group_id, pixels, label, and split.

The dataset includes train, validation, and test splits.

## Implementation
The pipeline validates the input schema, IDs, image dimensions, pixel values, labels, and split names.

It classifies the two supported synthetic image patterns and records predictions using stable record IDs.

## Robustness Property
Reversing the input record order must not change the prediction associated with any record ID.

Tests also verify that missing fields, duplicate IDs, invalid splits, incorrect image shapes, and unsupported image patterns are rejected.

## Run the pipeline
`python3 course-labs/r33-t019/pipeline.py`

## Run tests
`python3 -m unittest discover -s course-labs/r33-t019 -p "test_*.py" -v`

## Limitations
This is a deterministic, synthetic, CPU-only input-validation smoke test. It does not train or compile a Keras neural network and does not claim real-world model accuracy.

## Notebook Reference
See [10_neural_nets_with_keras.ipynb](../../10_neural_nets_with_keras.ipynb).

The implementation uses Python's standard library and requires no external dataset or paid service.
