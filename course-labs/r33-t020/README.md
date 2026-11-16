# R33-T020: Saved Text-Pipeline Interface

## Overview

This lab demonstrates a small text-classification pipeline that can be trained, saved, reloaded, and used for schema-validated predictions.

The example uses a small fictional dataset and a standard-library Multinomial Naive Bayes implementation. It demonstrates pipeline persistence and interface design, not a bidirectional RNN.

## Dataset

| ID | Text | Label |
|---|---|---:|
| t1 | red apple | 1 |
| t2 | green apple | 1 |
| t3 | blue sky | 0 |
| t4 | dark sky | 0 |
| t5 | red fruit | 1 |
| t6 | blue cloud | 0 |

## Data splits

- Training: t1, t2, t3, t4
- Validation: t5
- Test: t6

Validation and test records are kept separate from training.

## Pipeline interface

`pipeline.py` provides functionality to train the model, validate its schema, save it as JSON, load it again, and make predictions.

The saved model uses JSON rather than pickle.

## Run the tests

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r33-t020 -p "test_*.py" -v
```

## Limitations

This is a small synthetic demonstration. Its predictions are not evidence of real-world, clinical, or production classification performance. It demonstrates a saved text- pipeline interface, not a trained bidirectional RNN.
