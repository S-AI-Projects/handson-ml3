# Course Lab R33-T045: Image Preprocessing Contract & Channel Scaling

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T045 (Task 5 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T045`  
**Target Item:** `14_deep_computer_vision_with_cnns.ipynb` (`# CNN Architectures`)

---

## 1. Scope & Objective
Normalizes pixel intensities from $[0, 255]$ integer space to $[0.0, 1.0]$ float32 space and introduces an explicit channel dimension $(4, 4, 1)$. Enforces strict validation to prevent dimensions mismatch and data contamination.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The six 4x4 image samples in `fixture.json` are synthetic test inputs created solely for pipeline channel scaling verification.

---

## 2. Invariants
- Shape: $(4, 4) \implies (4, 4, 1)$
- Range: $[0, 255] \implies [0.0, 1.0]$
- Rejection of invalid sizes ($3\times 4, 4\times 5$), negative values, values $> 255$, and NaN/Inf.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t045 -p "test_*.py"
```
