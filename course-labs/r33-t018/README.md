# R33-T018: Reinforcement Learning Error Analysis

## Objective
This lab demonstrates error analysis for a simulated three-state reinforcement-learning environment.

## Environment
- States: 0, 1 and 2.
- Right advances one state.
- Left retreats one state, with a floor of 0.
- State 2 is terminal and gives reward 1.
- Episodes have a five-step cap.

## Data contract
Each transition includes:
- record_id
- episode_id
- state
- action
- next_state
- reward
- terminated
- predicted_value

The fixture has explicit train, validation and test splits. All data is synthetic and simulated.

## Error analysis
The pipeline compares predicted returns against expected returns.

The deliberately incorrect prediction is:
- Record ID: val-03
- Expected value: 0
- Predicted value: 0.75
- Reason: overestimated_zero_return

## Run the pipeline
python3 course-labs/r33-t018/pipeline.py

## Run tests
python3 -m unittest discover -s course-labs/r33-t018 -p "test_*.py" -v

## Limitations
This is a simulated example, not a trained Deep Q-Network. Predictions are supplied in the fixture to demonstrate error reporting.
