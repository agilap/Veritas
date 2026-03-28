"""
Layer 4 - Open-web RAG (staged)
Current stage: check-worthiness gate only.
"""

from functools import lru_cache
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
POST_TYPES = ["FACTUAL_CLAIM", "OPINION", "SATIRE", "LIFESTYLE", "PERSONAL", "OTHER"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TINYLLAMA_AVAILABLE = True
_TINYLLAMA_ERROR = None


@lru_cache(maxsize=1)
def _load_tinyllama():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


try:
    _load_tinyllama()
except Exception as exc:
    TINYLLAMA_AVAILABLE = False
    _TINYLLAMA_ERROR = str(exc)
    print(f"[Layer4] TinyLlama gate unavailable: {exc}")


def _extract_post_type(text: str) -> str:
    upper = text.upper()
    for post_type in POST_TYPES:
        if post_type in upper:
            return post_type
    return "FACTUAL_CLAIM"


def _check_worthiness(caption: str) -> Dict[str, object]:
    prompt = f"""Classify this social media post into exactly one category.
Post: "{caption}"
Categories:
- FACTUAL_CLAIM: verifiable statement about events, statistics, quotes, or facts
- OPINION: personal view or belief
- SATIRE: humor, parody, or irony
- LIFESTYLE: food, travel, fashion, fitness, personal life
- PERSONAL: personal story or experience
- OTHER: anything else
Respond with the category name only."""

    if not TINYLLAMA_AVAILABLE:
        post_type = "FACTUAL_CLAIM"
        return {"checkable": True, "post_type": post_type}

    try:
        tokenizer, model = _load_tinyllama()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        input_device = model.device if hasattr(model, "device") else torch.device(DEVICE)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(prompt):] if len(decoded) > len(prompt) else decoded
        post_type = _extract_post_type(response_text)
    except Exception:
        post_type = "FACTUAL_CLAIM"

    checkable = post_type == "FACTUAL_CLAIM"
    return {"checkable": checkable, "post_type": post_type}


def cross_reference(caption: str) -> dict:
    gate = _check_worthiness(caption)
    if not gate["checkable"]:
        return {
            "checkable": False,
            "post_type": gate["post_type"],
            "corroboration_score": None,
            "sources": [],
            "error": None,
        }

    return {
        "checkable": True,
        "post_type": gate["post_type"],
        "corroboration_score": None,
        "sources": [],
        "error": "RAG pipeline not implemented yet",
    }
