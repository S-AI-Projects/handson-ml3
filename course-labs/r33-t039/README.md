# R33-T039: Test an Input Perturbation

**Task:** R33-T039 · 9 of 10  
**Student:** Snehakumari Vinod Singh  
**Section:** 16 nlp with rnns and attention / An Encoder–Decoder Network for Neural Machine Translation  
**Notebook:** `16_nlp_with_rnns_and_attention.ipynb`  
**Branch:** `student/e26b07f0590/R33-T039`

---

## Goal

Test a stated robustness property for the text interface:
- **Valid Transformation:** Record-order permutation preserves documented predictions after ID alignment.
- **Normalization Property:** Case folding and whitespace padding perturbation preserves exact token sequence and model scoring.
- **Explicit Fallback:** Unseen out-of-vocabulary tokens map explicitly to an `<unk>` token without crashing.
- **Rejection of Incompatible Inputs:** Missing fields (`record_id`, `text`) or non-string inputs are strictly rejected with `ValueError`.

*Note:* This test asserts interface contract robustness and does not claim broad adversarial robustness.

---

## Synthetic Fixture

> **⚠️ Disclaimer:** These records are **fully synthetic** — generated for text interface perturbation testing only. Results here are **not** evidence of real-world, clinical, or production performance and must not be cited as such.

Six fictional records with IDs `t1..t6`:

| record_id | text        | label | Partition |
|-----------|-------------|-------|-----------|
| t1        | red apple   | 1     | Train     |
| t2        | green apple | 1     | Train     |
| t3        | blue sky    | 0     | Train     |
| t4        | dark sky    | 0     | Train     |
| t5        | red fruit   | 1     | Test      |
| t6        | blue cloud  | 0     | Test      |

---

## Robustness & Perturbation Properties

1. **Permutation Invariance with Stable IDs:**
   Evaluating records in permuted order (e.g. `[t6, t1, t4, t2, t5, t3]`) yields bit-for-bit identical predictions when aligned by `record_id`.
2. **Text Formatting Normalization:**
   Formatting perturbations (e.g., `"   RED APPLE   \t"`) normalize to `["red", "apple"]`.
3. **Explicit `<unk>` Fallback:**
   Tokens not present in training vocabulary map directly to index `1` (`<unk>`), maintaining numerical stability with neutral score contribution.
4. **Strict Schema Rejection:**
   Records missing required fields (`text`, `record_id`) or receiving non-string payloads immediately raise `ValueError`.

---

## Running Checks

Run unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t039 -p "test_*.py" -v
```

Run verification runner:
```bash
python3 work/R33-T039/run_verification.py
```

---

## Limits & Data Contract

- Standard library only — no heavy external deep learning frameworks required for interface contract verification.
- Validates text interface contracts rather than real-world machine translation BLEU scores.

## Optional Real-Data Extension

Any optional real-world translation extension must record:
- Source parallel corpus name, language pair, and license
- Vocabulary size and subword tokenization model (BPE / SentencePiece)
- Evaluation metric (BLEU / chrF) and test split isolation
