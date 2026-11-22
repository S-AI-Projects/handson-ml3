# Course Lab R33-T043: Minimal Majority-Class Baseline

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T043 (Task 3 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T043`  
**Target Item:** `11_training_deep_neural_networks.ipynb` (`## Nonsaturating Activation Functions`)

---

## 1. Scope & Objective
Implements the simplest honest benchmark baseline (majority-class classifier) for 4x4 image fixtures without learning weights.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The six 4x4 grayscale images in `fixture.json` (3 all-zero, 3 diagonal) are synthetic smoke-test representations. They do not represent real MNIST or CIFAR image datasets.

---

## 2. Baseline Protocol
- Computes mode of training labels.
- Emits constant predictions.
- Evaluates balanced accuracy offline on CPU.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t043 -p "test_*.py"
```
