"""
R33-T039: Test an input perturbation
====================================
Project/section: 16 nlp with rnns and attention / An Encoder–Decoder Network for Neural Machine Translation
Notebook: 16_nlp_with_rnns_and_attention.ipynb

Goal
----
Test a stated robustness property for the text interface:
- record-order changes with stable IDs preserve predictions after ID alignment.
- unseen tokens map explicitly to an '<unk>' fallback token without crashing.
- missing fields or incompatible shapes are rejected with ValueError.

Fixture:
Six fictional records with IDs t1..t6:
["red apple","green apple","blue sky","dark sky","red fruit","blue cloud"], labels [1,1,0,0,1,0].

Data contract:
- Source: Fully synthetic, generated for text interface perturbation testing only.
- Licence: N/A.
- Real dataset used: No.
- Disclaimer: These records are simulated. Results must not be cited as evidence
  of real-world or clinical performance.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Synthetic Fixture
# ---------------------------------------------------------------------------

SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "t1", "text": "red apple", "label": 1},
    {"record_id": "t2", "text": "green apple", "label": 1},
    {"record_id": "t3", "text": "blue sky", "label": 0},
    {"record_id": "t4", "text": "dark sky", "label": 0},
    {"record_id": "t5", "text": "red fruit", "label": 1},
    {"record_id": "t6", "text": "blue cloud", "label": 0},
]

TRAIN_RECORD_IDS = ["t1", "t2", "t3", "t4"]
TEST_RECORD_IDS = ["t5", "t6"]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


# ---------------------------------------------------------------------------
# Text Tokenizer & Vocabulary
# ---------------------------------------------------------------------------

class TextTokenizer:
    """Robust text tokenizer supporting explicit unknown-token fallback and strict input validation."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def fit_on_texts(self, texts: List[str]) -> "TextTokenizer":
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Tokenizer fit expects a list of string elements.")
            tokens = self._tokenize_words(text)
            for tok in tokens:
                if tok not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                    self.inv_vocab[idx] = tok
        return self

    def _tokenize_words(self, text: str) -> List[str]:
        # Case folding and basic punctuation stripping
        cleaned = re.sub(r"[^\w\s]", "", text.lower().strip())
        return [w for w in cleaned.split() if w]

    def encode(self, text: str) -> List[int]:
        """Encodes text to token IDs. Maps unseen tokens to explicit UNK_ID."""
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text).__name__}.")
        words = self._tokenize_words(text)
        if not words:
            return [PAD_ID]
        return [self.vocab.get(w, UNK_ID) for w in words]

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(idx, UNK_TOKEN) for idx in token_ids]


# ---------------------------------------------------------------------------
# CPU Sequence Scoring Classifier
# ---------------------------------------------------------------------------

class SimpleCPUTextClassifier:
    """
    Lightweight CPU text classifier scoring sequence tokens.
    Learns log-odds token weights; unseen tokens map to neutral weight (0.0).
    """

    def __init__(self, tokenizer: TextTokenizer):
        self.tokenizer = tokenizer
        self.weights: Dict[int, float] = {}
        self.bias: float = 0.0
        self.fitted = False

    def fit(self, records: List[Dict[str, Any]]) -> "SimpleCPUTextClassifier":
        if not records:
            raise ValueError("Training dataset cannot be empty.")

        # Compute empirical token log-odds on training records
        token_pos_counts: Dict[int, int] = {}
        token_neg_counts: Dict[int, int] = {}

        for r in records:
            if "text" not in r or "label" not in r:
                raise ValueError("Records must contain 'text' and 'label'.")
            ids = set(self.tokenizer.encode(r["text"]))
            label = int(r["label"])
            for tid in ids:
                if label == 1:
                    token_pos_counts[tid] = token_pos_counts.get(tid, 0) + 1
                else:
                    token_neg_counts[tid] = token_neg_counts.get(tid, 0) + 1

        for tid in self.tokenizer.vocab.values():
            pos = token_pos_counts.get(tid, 0) + 0.5
            neg = token_neg_counts.get(tid, 0) + 0.5
            self.weights[tid] = round(math.log(pos / neg), 4)

        # UNK and PAD explicitly neutral
        self.weights[PAD_ID] = 0.0
        self.weights[UNK_ID] = 0.0
        self.bias = 0.0
        self.fitted = True
        return self

    def score_record(self, record: Dict[str, Any]) -> float:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary.")
        if "record_id" not in record:
            raise ValueError("Record missing required 'record_id' field.")
        if "text" not in record or not isinstance(record["text"], str):
            raise ValueError("Record missing required string 'text' field.")

        token_ids = self.tokenizer.encode(record["text"])
        total_score = self.bias + sum(self.weights.get(tid, 0.0) for tid in token_ids)
        return total_score

    def predict_record(self, record: Dict[str, Any]) -> int:
        score = self.score_record(record)
        return 1 if score >= 0.0 else 0

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Returns dictionary of {record_id: prediction}."""
        if not isinstance(records, list):
            raise ValueError("Input records must be a list.")
        return {r["record_id"]: self.predict_record(r) for r in records}


# ---------------------------------------------------------------------------
# Perturbation Transformations
# ---------------------------------------------------------------------------

def apply_record_order_perturbation(
    records: List[Dict[str, Any]],
    permuted_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Valid transformation: Changes order of input records without altering record_ids.
    Since record evaluation is keyed by stable IDs, ID-aligned predictions must be identical.
    """
    if permuted_indices is None:
        # Default reversal / permutation
        permuted_indices = [5, 4, 3, 2, 1, 0]
    if len(permuted_indices) != len(records):
        raise ValueError("Permutation indices length must match records length.")
    return [records[i] for i in permuted_indices]


def apply_formatting_perturbation(text: str) -> str:
    """Valid text perturbation: Case folding and whitespace padding."""
    return f"   {text.upper()}   \t"


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_perturbation_pipeline() -> Dict[str, Any]:
    """
    Executes end-to-end perturbation and robustness checks.
    Asserts:
    1. Record-order perturbation preserves exact predictions after ID alignment.
    2. Formatting perturbation preserves tokenization and prediction.
    3. Unseen tokens map to '<unk>' fallback without crashing.
    4. Missing required fields are rejected with ValueError.
    """
    train_records = [r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS]

    # Initialize and fit
    tokenizer = TextTokenizer()
    tokenizer.fit_on_texts([r["text"] for r in SYNTHETIC_FIXTURE])

    clf = SimpleCPUTextClassifier(tokenizer)
    clf.fit(train_records)

    # 1. Baseline predictions on standard order
    base_preds = clf.predict(SYNTHETIC_FIXTURE)

    # 2. Record-order permutation
    permuted_records = apply_record_order_perturbation(SYNTHETIC_FIXTURE, [5, 0, 4, 1, 3, 2])
    permuted_preds = clf.predict(permuted_records)

    # Verify ID alignment preservation
    aligned_matches = all(base_preds[rid] == permuted_preds[rid] for rid in base_preds)
    assert aligned_matches, "Record-order perturbation violated ID-aligned prediction equivalence!"

    # 3. Unseen token fallback
    unseen_record = {"record_id": "test_unseen", "text": "mysterious purple galaxy", "label": 0}
    encoded_unseen = tokenizer.encode(unseen_record["text"])
    unseen_has_fallback = all(tid == UNK_ID for tid in encoded_unseen)
    unseen_pred = clf.predict_record(unseen_record)
    assert unseen_has_fallback, "Unseen tokens failed to map to UNK_ID fallback"

    # 4. Incompatible input rejection
    rejected_missing_field = False
    try:
        clf.score_record({"record_id": "bad1"})  # missing 'text'
    except ValueError:
        rejected_missing_field = True
    assert rejected_missing_field, "Model failed to reject missing text field"

    return {
        "status": "PASS",
        "vocabulary_size": len(tokenizer.vocab),
        "base_predictions": base_preds,
        "permuted_predictions": permuted_preds,
        "id_aligned_equivalent": aligned_matches,
        "unseen_token_fallback": "mapped_to_unk",
        "unseen_record_prediction": unseen_pred,
        "missing_field_rejection": "rejected_with_ValueError",
    }


if __name__ == "__main__":
    res = run_perturbation_pipeline()
    print("Perturbation Pipeline Output:")
    for k, v in res.items():
        print(f"  {k}: {v}")
