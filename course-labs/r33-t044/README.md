# Course Lab R33-T044: Evaluation Metrics & RL Chain Environment

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T044 (Task 4 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T044`  
**Target Item:** `18_reinforcement_learning.ipynb` (`# Policy Gradients`)

---

## 1. Scope & Objective
Implements a 3-state chain RL environment and independently verifies evaluation metrics (precision/recall zero-denominator behavior and regression MAE).

> ### ⚠️ Synthetic Pipeline Verification Notice
> The 3-state chain environment and synthetic metric vectors are pipeline verification benchmarks. They do not represent OpenAI Gym or real Atari environments.

---

## 2. Invariants
- `[right, right]` reaches terminal state 2 with entry reward 1.
- 5 `left` actions retreat with floor 0 and terminate at step cap with reward 0.
- Classification metric test: truth `[1,1,0,0]`, pred `[1,0,1,0]` $\implies$ precision=recall=0.5.
- Regression metric test: truth `[1,2,3]`, pred `[1,3,2]` $\implies$ MAE=2/3.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t044 -p "test_*.py"
```
