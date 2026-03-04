"""
Layer 1 — Text Fake News Classifier
Fine-tunes DistilBERT on the LIAR dataset from HuggingFace.
Run `python layers/text_classifier.py --train` once to produce the model checkpoint,
then inference is instant on every subsequent run.
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


# ── training helper ────────────────────────────────────────────────────────────
def train(epochs: int = 3, batch_size: int = 32):
    """
    Fine-tune DistilBERT on LIAR.
    Usage:  Open train.ipynb in Google Colab and run all cells
    or:     python layers/text_classifier.py --train
    """
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer
    from torch.utils.data import Dataset as TorchDataset

    print("⏳ Loading LIAR dataset from HuggingFace…")
    raw = load_dataset("liar")  # train / validation / test splits

    # LIAR label column is named "label" and is already an int 0-5
    LIAR_INT_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}  # collapse to 3 classes

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class LiarDataset(TorchDataset):
        def __init__(self, split):
            self.data = raw[split]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # LIAR 'statement' column contains the claim text
            enc = tokenizer(
                item["statement"],
                truncation=True, max_length=256, padding="max_length"
            )
            return {
                **{k: torch.tensor(v) for k, v in enc.items()},
                "labels": torch.tensor(LIAR_INT_MAP[item["label"]], dtype=torch.long),
            }

    train_ds = LiarDataset("train")
    val_ds   = LiarDataset("validation")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    print("🚀 Training DistilBERT on LIAR…")
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        train()
    else:
        # Quick inference smoke-test
        label, conf, tip = classify_text("The moon landing was faked by NASA in 1969.")
        print(f"Label: {label}  Confidence: {conf:.1%}\nTip: {tip}")
