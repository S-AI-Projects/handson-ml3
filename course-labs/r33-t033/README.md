# Course Lab R33-T033: Reinforcement Learning Three-State Chain & Minimal Baseline

**Student:** Snehakumari Vinod Singh  
**Task Code:** R33-T033 (Task 3 of 10)  
**Assigned Branch:** `student/e26b07f0590/R33-T033`  
**Related Issue:** [Issue #33](https://github.com/S-AI-Projects/handson-ml3/issues/33)  
**Target Item:** `18_reinforcement_learning.ipynb` (Neural Network Policies)

---

## 1. Scope & Objective

This contribution implements a self-contained, CPU-only simulated three-state Markov chain environment and minimal honest baseline comparison policy (`AlwaysRightPolicy`) for reinforcement learning contracts in `18_reinforcement_learning.ipynb`. It establishes clear lower-bound performance criteria, explicit step-cap handling, and strict transition schema validation before deploying complex neural network policies.

> ### ⚠️ Synthetic Environment Disclaimer
> The three-state chain environment and synthetic transition fixtures in `fixture.json` and `fixture.csv` are designed strictly for CPU smoke testing and baseline comparison.
>
> **These are NOT observations from any named real-world reinforcement learning benchmark (e.g., OpenAI Gym / Gymnasium CartPole or Atari).** They are pipeline verification fixtures and must never be cited as evidence of real-world or clinical predictive performance.

---

## 2. Three-State Chain Environment Dynamics

The environment (`ThreeStateChainEnv`) defines a bounded 3-state discrete MDP:
- **States:** `0` (Initial), `1` (Intermediate), `2` (Terminal).
- **Actions:**
  - `'right'`: Advances current state by $+1$, clamped at terminal state `2`.
  - `'left'`: Retreats current state by $-1$, floored at initial state `0`.
- **Rewards & Termination:**
  - Reaching state `2` yields terminal reward $+1.0$ and terminates the episode immediately.
  - Step cap: $5$ actions. If step cap is reached without reaching state `2`, the episode terminates with reward $0.0$.
- **Concrete Trajectories:**
  - `['right', 'right']`: Starts at `0`, traverses `0 -> 1 -> 2`, ends with total reward $1.0$ in $2$ steps.
  - $5$ `'left'` actions: Traverses `0 -> 0 -> 0 -> 0 -> 0 -> 0`, hits step cap of $5$, ends with total reward $0.0$.

---

## 3. Minimal Baseline Comparison

| Policy | Action Strategy | Mean Reward (Held-out) | Mean Steps | Success Rate |
| :--- | :--- | :--- | :--- | :--- |
| **AlwaysRightPolicy** (Honest Baseline) | Always chooses `'right'` | **1.0** | **2.0** | **100%** |
| **AlwaysLeftPolicy** (Negative Control) | Always chooses `'left'` | 0.0 | 5.0 | 0% |
| **RandomPolicy** (Uniform Random) | Chooses `'left'` or `'right'` randomly | Variable ($< 1.0$) | $> 2.0$ | $< 100\%$ |

---

## 4. Transition Schema Specification

Each transition record adheres to the following contract:

| Field | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `episode_id` | `str` | Non-empty string | Unique trajectory/episode identifier. |
| `state` | `int` | `{0, 1, 2}` | Current environment state before action. |
| `action` | `str` | `{'right', 'left'}` | Selected discrete action. |
| `next_state` | `int` | `{0, 1, 2}` | Resulting state after action execution. |
| `reward` | `float` | `{0.0, 1.0}` | Transition reward ($1.0$ only upon entering state 2). |
| `terminated` | `bool` | `{True, False}` | Flag indicating episode completion. |

---

## 5. Running Automated Checks & Tests

### Run Unit Tests
```bash
python3 -m unittest discover -s course-labs/r33-t033 -p "test_*.py" -v
```

### Run Verification Runner (Tool versions, evidence, exit code 0)
```bash
python3 work/R33-T033/run_verification.py
```
Outputs:
- Generates `work/R33-T033/tool_versions.json`
- Generates `work/R33-T033/unittest_output.txt`
- Generates `work/R33-T033/evidence.json`
- Exits with code `0` on success or non-zero on assertion failure.
