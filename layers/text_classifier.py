"""
Layer 1 — Text Fake News Classifier
Fine-tunes DistilBERT on LIAR + Philippines fact-check data.
Training is done in train.ipynb (Colab notebook).
This file handles inference only at runtime.
"""

import os
import torch
import numpy as np
from typing import Tuple

# ── inference path (always imported) ──────────────────────────────────────────
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "liar_distilbert")

# Maps LIAR's 6-class label → 3-class credibility bucket
LIAR_LABEL_MAP = {
    "pants-fire": 0,  # False
    "false":      0,
    "barely-true": 1, # Uncertain
    "half-true":  1,
    "mostly-true": 2, # Credible
    "true":       2,
}
ID2CRED   = {0: "❌ Likely False", 1: "⚠️ Uncertain", 2: "✅ Likely Credible"}
CRED_TIPS = {
    0: "The text contains patterns strongly associated with misinformation.",
    1: "The text contains mixed signals — treat with caution.",
    2: "The text reads as credible, though always verify independently.",
}


def load_model():
    """Load fine-tuned model (or fall back to base DistilBERT if not trained yet)."""
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    if os.path.isdir(MODEL_DIR):
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        # Untrained fallback — labels are random but pipeline still runs
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
    model.eval()
    return tokenizer, model


_tokenizer, _model = None, None  # lazy-loaded


def classify_text(caption: str) -> Tuple[str, float, str]:
    """
    Returns:
        label   — human-readable credibility label
        conf    — confidence 0-1
        tip     — one-sentence explanation
    """
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer, _model = load_model()

    inputs = _tokenizer(
        caption, return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred   = int(np.argmax(probs))
    conf   = float(probs[pred])
    return ID2CRED[pred], round(conf, 3), CRED_TIPS[pred]


if __name__ == "__main__":
    # Quick inference smoke-test
    label, conf, tip = classify_text("The moon landing was faked by NASA in 1969.")
    print(f"Label: {label}  Confidence: {conf:.1%}\nTip: {tip}")
