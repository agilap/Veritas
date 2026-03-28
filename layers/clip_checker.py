"""
Layer 2 — Caption ↔ Image Consistency (CLIP)
Uses OpenAI CLIP ViT-B/32 — zero fine-tuning needed.
Loads calibrated thresholds from models/clip_thresholds.json if available.
Returns cosine similarity and a human flag.
"""

import torch
import json
import os
import numpy as np
from PIL import Image
from typing import Any, Dict

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

_MODEL_ID = "openai/clip-vit-base-patch32"
_clip_model, _clip_processor = None, None  # lazy-loaded
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

BLIP_AVAILABLE = True
_blip_processor, _blip_model = None, None
try:
    _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    _blip_model.to(device)
    _blip_model.eval()
except Exception as _blip_load_error:
    BLIP_AVAILABLE = False
    _blip_processor = None
    _blip_model = None
    print(f"⚠️ BLIP unavailable: {_blip_load_error}")


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


# Load calibrated thresholds if available, otherwise use MS-COCO defaults
_THRESH_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "clip_thresholds.json")

def _load_thresholds():
    defaults = {"match": 0.28, "uncertain": 0.20}
    if os.path.isfile(_THRESH_FILE):
        try:
            with open(_THRESH_FILE) as f:
                data = json.load(f)
            print(f"✅ Loaded calibrated CLIP thresholds: match={data['match']}, uncertain={data['uncertain']}")
            return {"match": data["match"], "uncertain": data["uncertain"]}
        except Exception as e:
            print(f"⚠️ Could not load {_THRESH_FILE}: {e} — using defaults")
    return defaults

THRESHOLDS = _load_thresholds()


def blip_verify(image: Image.Image) -> Dict[str, Any]:
    if not BLIP_AVAILABLE:
        return {
            "blip_caption": None,
            "text_text_similarity": None,
            "error": "BLIP unavailable — CLIP-only mode",
        }

    try:
        image_rgb = image.convert("RGB")
        inputs = _blip_processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _blip_model.generate(**inputs, max_new_tokens=40)
        blip_caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True).strip()

        original_caption = str(image.info.get("_user_caption", "")).strip()
        if not original_caption:
            return {
                "blip_caption": blip_caption or None,
                "text_text_similarity": None,
                "error": "Original caption unavailable for BLIP text-to-text similarity",
            }

        user_text_vec = embed_text(original_caption)
        blip_text_vec = embed_text(blip_caption)
        text_text_similarity = _cosine(user_text_vec, blip_text_vec)

        return {
            "blip_caption": blip_caption,
            "text_text_similarity": float(round(text_text_similarity, 4)),
            "error": None,
        }
    except Exception as exc:
        return {
            "blip_caption": None,
            "text_text_similarity": None,
            "error": f"BLIP verification failed safely: {exc}",
        }


def check_caption_image(caption: str, image: Image.Image) -> Dict[str, Any]:
    """
    Returns a dict with CLIP and BLIP consistency signals.
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

    # Pass original caption through image metadata to honor blip_verify(image) signature.
    image.info["_user_caption"] = caption
    blip_result = blip_verify(image)

    return {
        "similarity": round(sim, 4),
        "flag": flag,
        "explanation": exp,
        "blip_caption": blip_result.get("blip_caption"),
        "text_text_similarity": blip_result.get("text_text_similarity"),
        "error": blip_result.get("error"),
    }


if __name__ == "__main__":
    # Quick smoke-test with a generated solid-colour image
    img   = Image.new("RGB", (224, 224), color=(30, 100, 200))
    result = check_caption_image("A vast ocean under a stormy sky", img)
    print(
        f"Similarity: {result['similarity']:.4f}\n"
        f"{result['flag']}\n"
        f"{result['explanation']}\n"
        f"BLIP caption: {result['blip_caption']}\n"
        f"BLIP text-text sim: {result['text_text_similarity']}\n"
        f"BLIP error: {result['error']}"
    )
