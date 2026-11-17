# Course Lab R33-T009: Input Perturbation Robustness for Gradient Boosted Classifier

**Student:** NAKSHATRA MESHRAM  
**Task:** R33-T009 (9 of 10)  
**Notebook Item:** `07_ensemble_learning_and_random_forests.ipynb`  
**Target Section:** `## Gradient Boosting`  
**Repository:** [S-AI-Projects/handson-ml3](https://github.com/S-AI-Projects/handson-ml3)

---

## 1. Overview & Objective

In Chapter 07 (*Ensemble Learning and Random Forests*), the **Gradient Boosting** section introduces sequential ensemble boosting on pseudo-residuals.

This lab implements a bounded robustness evaluation testing the classification interface against input perturbations:
1. **Valid Transformation (Permutation Invariance with Stable IDs):** Shuffling or reordering input records within an inference batch preserves identical prediction labels and probabilities when aligned by `record_id`:
   $$\hat{y}_{\text{permuted}}[record\_id] == \hat{y}_{\text{original}}[record\_id], \quad P(y=1|x)_{\text{permuted}}[record\_id] == P(y=1|x)_{\text{original}}[record\_id]$$
2. **Unsupported Input Rejection:** Rejects incompatible input shapes (e.g. 1 or 3 features instead of 2) with `IncompatibleShapeError`, and non-finite values (`NaN`, `Inf`) with `InvalidFeatureValueError`.
3. **Bounded Scope Statement:** This test demonstrates structural batch-order invariance and shape schema rejection; it explicitly **does not claim broad adversarial robustness** against arbitrary continuous perturbation vectors.

> **Notice:** The 8-row synthetic classification fixture is designed exclusively for **pipeline verification** and interface robustness testing. It is not an empirical substitute for real-world benchmark datasets.

---

## 2. Gradient Boosted Stump Model

The classification model (`GradientBoostedStumpClassifier`) uses sequential boosting on binary cross-entropy pseudo-residuals:
- **Base Learner:** Decision stumps splitting on feature thresholds $\theta$.
- **Split Criterion:** Minimizes the residual sum of squares (RSS / variance reduction) on negative gradients $r_i = y_i - \sigma(F_m(x_i))$.
- **Leaf Updates:** Newton-Raphson step $\gamma = \frac{\sum r_i}{\sum p_i (1 - p_i)}$.
- **Inference:** $P(y=1|x) = \frac{1}{1 + \exp(-F_M(x))}$; predicted class $\hat{y} = 1$ if $P \ge 0.5$, else $0$.

---

## 3. Synthetic Fixture & Stratified Split

Eight synthetic rows with $x_1 \in [0..7]$, $x_2 = 1.0$, and binary labels:
- Class `0` ($x_1 \in \{0, 1, 2, 3\}$, Group `g0`).
- Class `1` ($x_1 \in \{4, 5, 6, 7\}$, Group `g1`).

### Stratified Partitioning:
- **Train (4 rows):** `r1` ($x_1=0$), `r4` ($x_1=3$), `r5` ($x_1=4$), `r8` ($x_1=7$) — 50% class 0, 50% class 1.
- **Validation (2 rows):** `r2` ($x_1=1$), `r6` ($x_1=5$) — 50% class 0, 50% class 1.
- **Test (2 rows):** `r3` ($x_1=2$), `r7` ($x_1=6$) — 50% class 0, 50% class 1.

---

## 4. Verification Commands

Run unit test discovery:
```bash
python3 -m unittest discover -s course-labs/r33-t009 -p "test_*.py" -v
```

Run comprehensive verification runner:
```bash
python3 work/R33-T009/run_verification.py
```
