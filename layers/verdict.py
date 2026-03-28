"""
Layer 5 — Plain-English Verdict (TinyLlama)
Receives structured scores from Layers 1-4 → outputs a natural-language verdict.

TinyLlama is loaded once in Layer 4 and reused here to avoid duplicate VRAM usage.
Falls back to a rule-based verdict if model inference fails.
"""

import torch
from typing import Dict, Optional

from layers.source_checker import _load_tinyllama, TINYLLAMA_AVAILABLE


_VERDICT_TOKENIZER = None
_VERDICT_MODEL = None

if TINYLLAMA_AVAILABLE:
    try:
        _VERDICT_TOKENIZER, _VERDICT_MODEL = _load_tinyllama()
    except Exception:
        _VERDICT_TOKENIZER, _VERDICT_MODEL = None, None


# ── template fallback (no LLM needed for quick tests) ─────────────────────────

def _compute_text_score(text_label: str, text_conf: float) -> float:
    if "False" in text_label:
        return max(0.0, 1.0 - text_conf)
    if "Uncertain" in text_label:
        return 0.5 * text_conf
    return text_conf


def _non_checkable_verdict(post_type: str) -> str:
    mapping = {
        "OPINION": "NOT FACT-CHECKABLE (Opinion)",
        "SATIRE": "SATIRE DETECTED",
        "LIFESTYLE": "NOT FACT-CHECKABLE (Lifestyle content)",
        "PERSONAL": "NOT FACT-CHECKABLE (Personal story)",
        "OTHER": "NOT FACT-CHECKABLE",
    }
    verdict = mapping.get(post_type, "NOT FACT-CHECKABLE")
    return f"**{verdict}**\n\nThis post is classified as {post_type.lower()} and is not suitable for factual corroboration scoring."


def _rule_based_verdict(
    text_label: str,
    text_conf: float,
    clip_sim: float,
    blip_sim: float,
    corroboration: float,
    video_sim: float,
    n_sources: int,
    is_video: bool,
    l4: Optional[Dict[str, object]] = None,
) -> str:
    if l4 and not l4.get("checkable", True):
        post_type = str(l4.get("post_type", "OTHER"))
        return _non_checkable_verdict(post_type)

    weights = [0.25, 0.20, 0.15, 0.30, 0.10]
    text_score = _compute_text_score(text_label, text_conf)
    final_score = (
        weights[0] * text_score
        + weights[1] * clip_sim
        + weights[2] * blip_sim
        + weights[3] * corroboration
        + weights[4] * video_sim
    )

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
    if blip_sim < 0.3:
        issues.append(f"caption vs BLIP caption similarity is low ({blip_sim*100:.0f}%)")

    if final_score >= 0.70 and not issues:
        verdict = "✅ Likely Authentic"
        body    = "All signals are consistent — the caption appears credible and matches the visual content."
    elif final_score >= 0.40 and len(issues) <= 2:
        verdict = "⚠️ Suspicious"
        body    = (
            f"Mixed signals detected (composite score {final_score:.0%}). "
            + (f"Main concern: {issues[0]}." if issues else "Treat with caution.")
        )
    else:
        verdict = "🚨 Likely Fabricated"
        if issues:
            body = (
                f"Multiple red flags detected (composite score {final_score:.0%}): "
                + "; ".join(issues)
                + "."
            )
        else:
            body = f"Composite score is low ({final_score:.0%}) even though explicit red flags were limited."

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
    l4: Optional[Dict[str, object]] = None,
    blip_sim: Optional[float] = None,
    video_sim: Optional[float] = None,
) -> str:
    """
    Returns a plain-English verdict string (markdown-safe).
    Falls back to template if LLM is unavailable.
    """
    if l4 and not l4.get("checkable", True):
        post_type = str(l4.get("post_type", "OTHER"))
        return _non_checkable_verdict(post_type)

    blip_value = 0.5 if blip_sim is None else float(blip_sim)
    video_value = (clip_sim if is_video else 0.5) if video_sim is None else float(video_sim)

    if not use_llm or _VERDICT_MODEL is None or _VERDICT_TOKENIZER is None:
        return _rule_based_verdict(
            text_label=text_label,
            text_conf=text_conf,
            clip_sim=clip_sim,
            blip_sim=blip_value,
            corroboration=corroboration,
            video_sim=video_value,
            n_sources=n_sources,
            is_video=is_video,
            l4=l4,
        )

    weights = [0.25, 0.20, 0.15, 0.30, 0.10]
    text_score = _compute_text_score(text_label, text_conf)
    weighted_score = (
        weights[0] * text_score
        + weights[1] * clip_sim
        + weights[2] * blip_value
        + weights[3] * corroboration
        + weights[4] * video_value
    )

    media_type = "video" if is_video else "image"
    user_msg = f"""Analyze this social media post:

CAPTION: "{caption}"

ANALYSIS SCORES:
- Text credibility (DistilBERT on LIAR): {text_label} ({text_conf:.1%} confidence)
- Caption ↔ {media_type} consistency (CLIP): {clip_flag} (similarity={clip_sim*100:.1f}%)
- Caption ↔ generated caption consistency (BLIP text-text): {blip_value:.1%}
- External source corroboration: {corroboration:.1%} ({n_sources} sources checked)
- Weighted authenticity score: {weighted_score:.1%}

WEIGHTS:
- text: 0.25
- clip: 0.20
- blip: 0.15
- source: 0.30
- video: 0.10

Write a plain-English verdict (≤4 sentences). Start with the overall verdict: Likely Authentic / Suspicious / Likely Fabricated."""

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = _VERDICT_TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = _VERDICT_TOKENIZER(prompt, return_tensors="pt")
        model_device = _VERDICT_MODEL.device if hasattr(_VERDICT_MODEL, "device") else torch.device("cpu")
        model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

        with torch.no_grad():
            out = _VERDICT_MODEL.generate(
            **model_inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        generated = _VERDICT_TOKENIZER.decode(out[0], skip_special_tokens=True)
        if prompt in generated:
            generated = generated.split(prompt, 1)[1]
        return generated.strip()

    except Exception as e:
        err_text = str(e).lower()
        if "out of memory" in err_text and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[TinyLlama] Falling back to rule-based verdict: {e}")
        return _rule_based_verdict(
            text_label=text_label,
            text_conf=text_conf,
            clip_sim=clip_sim,
            blip_sim=blip_value,
            corroboration=corroboration,
            video_sim=video_value,
            n_sources=n_sources,
            is_video=is_video,
            l4=l4,
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
