"""
Layer 2 — Caption ↔ Image Consistency (CLIP)
Uses OpenAI CLIP ViT-B/32 from HuggingFace — zero fine-tuning needed.
Returns cosine similarity and a human flag.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple

from transformers import CLIPProcessor, CLIPModel

_MODEL_ID = "openai/clip-vit-base-patch32"
_clip_model, _clip_processor = None, None  # lazy-loaded


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("⏳ Loading CLIP ViT-B/32…")
        _clip_model     = CLIPModel.from_pretrained(_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(_MODEL_ID)
        _clip_model.eval()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def embed_text(text: str) -> np.ndarray:
    _load_clip()
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        feat = _clip_model.get_text_features(**inputs)
    return feat.squeeze().numpy()


def embed_image(image: Image.Image) -> np.ndarray:
    _load_clip()
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feat = _clip_model.get_image_features(**inputs)
    return feat.squeeze().numpy()


# Empirical thresholds calibrated on MS-COCO captions
THRESHOLDS = {
    "match":    0.28,   # similarity ≥ 0.28 → caption matches image
    "uncertain": 0.20,  # 0.20–0.28 → borderline
    # < 0.20 → mismatch
}


def check_caption_image(caption: str, image: Image.Image) -> Tuple[float, str, str]:
    """
    Returns:
        similarity  — cosine sim 0-1 (CLIP space)
        flag        — '✅ Match' | '⚠️ Uncertain' | '❌ Mismatch'
        explanation — plain-English sentence
    """
    t_vec = embed_text(caption)
    i_vec = embed_image(image)
    sim   = _cosine(t_vec, i_vec)
    pct   = round(sim * 100, 1)

    if sim >= THRESHOLDS["match"]:
        flag = "✅ Caption matches image"
        exp  = f"CLIP similarity is {pct}% — the caption appears consistent with the visual content."
    elif sim >= THRESHOLDS["uncertain"]:
        flag = "⚠️ Caption may not match image"
        exp  = f"CLIP similarity is {pct}% — borderline. The caption partially aligns but may be misleading."
    else:
        flag = "❌ Caption does NOT match image"
        exp  = f"CLIP similarity is only {pct}% — strong evidence the caption does not describe this image."

    return round(sim, 4), flag, exp


if __name__ == "__main__":
    # Quick smoke-test with a generated solid-colour image
    img   = Image.new("RGB", (224, 224), color=(30, 100, 200))
    sim, flag, exp = check_caption_image("A vast ocean under a stormy sky", img)
    print(f"Similarity: {sim:.4f}\n{flag}\n{exp}")
