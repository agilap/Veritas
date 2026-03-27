import os
from functools import lru_cache

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["TRUE", "HALF-TRUE", "FALSE"]


@lru_cache(maxsize=1)
def _load_text_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label={0: "TRUE", 1: "HALF-TRUE", 2: "FALSE"},
        label2id={"TRUE": 0, "HALF-TRUE": 1, "FALSE": 2},
        ignore_mismatched_sizes=True,
    )

    adapter_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "liar_lora")
    )

    if os.path.isdir(adapter_dir):
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_lin", "v_lin"],
            task_type=TaskType.SEQ_CLS,
        )
        peft_ready_model = get_peft_model(model, lora_config)
        model = PeftModel.from_pretrained(peft_ready_model, adapter_dir)

    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def analyze_text(caption: str) -> dict:
    try:
        tokenizer, model, device = _load_text_model()

        encoded = tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)

        confidence_tensor, prediction_tensor = torch.max(probabilities, dim=-1)
        prediction_index = int(prediction_tensor.item())
        label = LABELS[prediction_index] if prediction_index < len(LABELS) else "FALSE"

        return {
            "layer": 1,
            "label": label,
            "confidence": float(confidence_tensor.item()),
            "error": None,
        }
    except Exception as e:
        return {
            "layer": 1,
            "label": "UNKNOWN",
            "confidence": 0.0,
            "error": str(e),
        }