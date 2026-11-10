# R33-T051: Schema validator for Char-RNN text inputs

Task: validate `record_id`, `text`, `label` for the "Generating Shakespearean
Text Using a Character RNN" section of `16_nlp_with_rnns_and_attention.ipynb`.
This is the bounded component only, not a reproduction of the full project.

## Data contract
- `record_id`: non-empty string, unique
- `text`: non-empty string
- `label`: int, 0 or 1 (bool and float rejected)
- No other fields allowed

## Data
Six **synthetic** fictional records (t1..t6). This is a pipeline fixture, not a
real corpus. Results are pipeline verification only, not evidence of real-world
or clinical performance. Splits: train t1-t4, validation t5, test t6.

## Result
Valid fixture: 0 failures. Missing ID, malformed text/label and duplicate ID
each fail with the record ID, field and a specific reason. No record is
silently dropped.

## Limits
Checks structure only, not text quality. Offline, CPU-only, stdlib only.

## Optional real-data extension
Any real dataset must separately record source, licence, access requirements
and split design.

## Run
    python3 -m unittest discover -s course-labs/r33-t051 -p "test_*.py"
    python3 course-labs/r33-t051/check.py
