"""
Layer 5 — Plain-English Verdict (TinyLlama)
Receives structured scores from Layers 1-4 → outputs a natural-language verdict.

TinyLlama-1.1B-Chat is used for its tiny footprint (~600 MB) — perfect for HF Spaces CPU.
Falls back to a template-based verdict if the model can't be loaded (e.g., in unit tests).
"""

import os
import torch
from typing import Optional

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_pipe = None  # lazy-loaded


def _load_pipeline():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        print("⏳ Loading TinyLlama…")
        _pipe = pipeline(
            "text-generation",
            model=TINYLLAMA_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )


# ── template fallback (no LLM needed for quick tests) ─────────────────────────

def _template_verdict(
    text_label: str,
    text_conf: float,
    clip_sim: float,
    clip_flag: str,
    corroboration: float,
    n_sources: int,
    is_video: bool,
) -> str:
    issues = []
    if "False" in text_label or "Uncertain" in text_label:
        issues.append(f"the text reads as '{text_label}' ({text_conf:.0%} confidence)")
    if clip_sim < 0.20:
        media = "video" if is_video else "image"
        issues.append(f"the caption does not match the {media} (CLIP: {clip_sim*100:.0f}%)")
    elif clip_sim < 0.28:
        issues.append(f"the caption only partially matches the visual (CLIP: {clip_sim*100:.0f}%)")
    if corroboration < 0.4 and n_sources > 0:
        issues.append(f"only {corroboration:.0%} of sources corroborate the claim")
    elif n_sources == 0:
        issues.append("no external sources could be verified")

    if not issues:
        verdict = "✅ Likely Authentic"
        body    = "All signals are consistent — the caption appears credible and matches the visual content."
    elif len(issues) == 1:
        verdict = "⚠️ Suspicious"
        body    = f"One red flag found: {issues[0]}. Treat with caution."
    else:
        verdict = "🚨 Likely Fabricated"
        body    = "Multiple red flags detected: " + "; ".join(issues) + "."

    return f"**{verdict}**\n\n{body}"


# ── LLM-based verdict ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are TruthScan, a social media fact-checking AI.
You receive structured analysis scores for a post and write a concise, plain-English verdict.
Be direct. Never exceed 4 sentences. Always state the final verdict in the first sentence.
Do not use markdown except for bold on the verdict word."""

def generate_verdict(
    caption: str,
    text_label: str,
    text_conf: float,
    clip_sim: float,
    clip_flag: str,
    corroboration: float,
    n_sources: int,
    is_video: bool = False,
    use_llm: bool = True,
) -> str:
    """
    Returns a plain-English verdict string (markdown-safe).
    Falls back to template if LLM is unavailable.
    """
    if not use_llm:
        return _template_verdict(
            text_label, text_conf, clip_sim, clip_flag, corroboration, n_sources, is_video
        )

    media_type = "video" if is_video else "image"
    user_msg = f"""Analyze this social media post:

CAPTION: "{caption}"

ANALYSIS SCORES:
- Text credibility (DistilBERT on LIAR): {text_label} ({text_conf:.1%} confidence)
- Caption ↔ {media_type} consistency (CLIP): {clip_flag} (similarity={clip_sim*100:.1f}%)
- External source corroboration: {corroboration:.1%} ({n_sources} sources checked)

Write a plain-English verdict (≤4 sentences). Start with the overall verdict: Likely Authentic / Suspicious / Likely Fabricated."""

    try:
        _load_pipeline()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt = _pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = _pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        generated = out[0]["generated_text"]
        # Strip the prompt prefix
        if prompt in generated:
            generated = generated[len(prompt):]
        return generated.strip()

    except Exception as e:
        print(f"[TinyLlama] Falling back to template: {e}")
        return _template_verdict(
            text_label, text_conf, clip_sim, clip_flag, corroboration, n_sources, is_video
        )


if __name__ == "__main__":
    v = generate_verdict(
        caption="Massive flood hits Manila, Philippines 2024",
        text_label="⚠️ Uncertain",
        text_conf=0.61,
        clip_sim=0.18,
        clip_flag="❌ Caption does NOT match image",
        corroboration=0.3,
        n_sources=4,
        is_video=False,
        use_llm=False,  # template-only for smoke test
    )
    print(v)
