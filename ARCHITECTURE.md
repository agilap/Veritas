# Veritas v2 — ARCHITECTURE.md
> System design, data flow, model choices, and research grounding for TruthScan.

---

## 1. System Overview

```
User Input
   ├── Caption (text)
   └── Image OR Video file
         │
         ▼
┌──────────────────────────────────────────────────────┐
│                      PIPELINE                        │
│                                                      │
│  L1: DistilBERT ──► Credibility Score                │
│                                                      │
│  L2: CLIP ──────► Caption↔Image Similarity           │
│  L6: BLIP-2 ───► Auto-caption → CLIP re-score        │
│                                                      │
│  L3: OpenCV + CLIP ► Caption↔Video Score             │
│                      (keyframe avg + worst frame)    │
│                                                      │
│  L4: RAG Cross-Reference                             │
│   ├── L4a: Check-worthiness gate                     │
│   │        (is the post even fact-checkable?)        │
│   │   ├── NOT checkable → skip, label post type      │
│   │   └── IS checkable ──►                           │
│   ├── L4b: Open-web search (Serper / Tavily / DDG)   │
│   ├── L4c: Scrape + chunk top results                │
│   ├── L4d: TinyLlama stance per source               │
│   │        (SUPPORTS / REFUTES / IRRELEVANT)         │
│   └── L4e: Aggregate → corroboration_score           │
│                                                      │
│  L5: TinyLlama ─► Plain-English Verdict              │
│                                                      │
└──────────────────────────────────────────────────────┘
         │
         ▼
   Gradio UI Output
   ├── Verdict Banner (Likely Fake / Likely Real / Uncertain / Not Checkable)
   ├── Post type label (News Claim / Opinion / Satire / Lifestyle / etc.)
   ├── Score breakdown (each layer)
   ├── Suspicious frame (video mode)
   └── Evidence list with per-source stance + URL
```

---

## 2. Layer-by-Layer Design

### Layer 1 — Text Credibility (DistilBERT)

| Item | Detail |
|---|---|
| Model | `distilbert-base-uncased` fine-tuned on LIAR |
| Dataset | LIAR (12.8k statements, 6 classes → collapsed to 3: True / Half-True / False) |
| Output | `{ label: str, confidence: float }` |
| Research basis | Setiawan et al. (PLoS ONE 2025) confirm CLIP+DistilBERT multimodal combo achieves F1 86% on Fakeddit. Note: adding DistilBERT on top of CLIP adds marginal gain — keep it lightweight and use LoRA fine-tuning to avoid overfitting. |
| File | `layers/layer1_text.py` |

**Key design decision:** Use LoRA (rank=16) for fine-tuning per Setiawan et al. (2025) to prevent overfitting on limited labeled data.

---

### Layer 2 — Caption ↔ Image Consistency (CLIP)

| Item | Detail |
|---|---|
| Model | `ViT-B/32` via `open-clip-torch` (HuggingFace) |
| Method | Cosine similarity between text embedding and image embedding |
| Threshold | <0.25 = ⚠️ Mismatch; 0.25–0.45 = 🟡 Weak; >0.45 = ✅ Consistent |
| Output | `{ similarity: float, flag: str }` |
| Research basis | NewsCLIPpings benchmark (Luo et al., EMNLP 2021) is the gold-standard dataset for this exact task. WACV 2025 (Papadopoulos et al.) warns that CLIP alone exploits surface-level shortcuts — mitigated here by combining with BLIP-2 auto-captions in Layer 6. |
| File | `layers/layer2_clip.py` |

---

### Layer 3 — Caption ↔ Video Consistency (OpenCV + CLIP)

| Item | Detail |
|---|---|
| Method | Extract 1 keyframe per 2 seconds with `cv2.VideoCapture` |
| Scoring | Run CLIP on each frame → average score + identify worst frame |
| Output | `{ avg_similarity: float, worst_frame_idx: int, worst_frame_image: PIL.Image }` |
| Research basis | VMID (Zhong et al., arXiv 2411.10032, 2024) demonstrates keyframe-level multimodal fusion for short video misinformation detection. |
| File | `layers/layer3_video.py` |

---

### Layer 4 — RAG Cross-Reference (Open Web)

> **Design shift:** Wikipedia and Reddit are removed entirely. The entire open web is the evidence corpus, accessed via search APIs. Crucially, a check-worthiness gate fires first — most social media posts are not fact-checkable (opinions, humor, lifestyle, personal stories) and should never enter the retrieval pipeline.

---

#### L4a — Check-Worthiness Gate

| Item | Detail |
|---|---|
| Purpose | Classify whether the caption contains a verifiable factual claim before doing any web search |
| Method | Zero-shot prompt to TinyLlama: "Does this post make a verifiable factual claim? Classify as: FACTUAL_CLAIM / OPINION / SATIRE / LIFESTYLE / PERSONAL / OTHER" |
| Checkable types | Event/Property Claim, Numerical Claim, Quote Attribution, Causal Claim, Position Statement (from AVeriTeC taxonomy) |
| Non-checkable types | Personal opinion, humor/satire, lifestyle content, product posts, personal anecdotes |
| Output | `{ checkable: bool, post_type: str }` |
| Research basis | AVeriTeC (2024) establishes the 5 checkable claim types. Established automated fact-checking literature mandates claim detection as the mandatory first stage before any retrieval. |

**If `checkable == False`:** Layer 4 returns immediately with `{ checkable: false, post_type: str, corroboration_score: None }`. The pipeline continues to L5, which produces a special verdict: "This post does not appear to make a verifiable factual claim. Verdict: NOT FACT-CHECKABLE."

---

#### L4b — Open-Web Evidence Retrieval

| Item | Detail |
|---|---|
| Method | Extract 2–3 search queries from the caption → query web search API → retrieve top 5–8 result URLs |
| Primary API | **Serper.dev** (2,500 free searches/month) — used in 2025 LLM fact-checking research |
| Fallback API | **Tavily API** (free tier, built for RAG/AI research) |
| Last-resort | **DuckDuckGo Instant Answer API** (no key, no rate limit, limited) |
| Query generation | TinyLlama rewrites the caption as 2–3 neutral search queries designed to find confirming AND refuting evidence |
| Output | `{ queries: list[str], result_urls: list[str] }` |
| Research basis | 2025 study verified 17,856 real-world claims by querying a web search API for each claim and retrieving top 10 results for LLM reasoning. |

---

#### L4c — Scrape & Chunk Evidence

| Item | Detail |
|---|---|
| Method | For each URL from L4b: `requests.get()` + `BeautifulSoup` to extract body text, strip boilerplate |
| Chunking | Split into 300-token chunks, keep top 3 most relevant chunks per URL using TF-IDF cosine similarity against the original caption |
| Timeout | 5s per URL — skip silently if unreachable |
| Output | `{ evidence_chunks: list[{url, text, relevance_score}] }` |

---

#### L4d — Per-Source Stance Classification

| Item | Detail |
|---|---|
| Method | For each evidence chunk: TinyLlama prompt → classify stance as SUPPORTS / REFUTES / IRRELEVANT |
| Prompt | "Given this claim: '{caption}' and this evidence: '{chunk}' — does the evidence SUPPORT, REFUTE, or is it IRRELEVANT to the claim? Answer with one word only." |
| Output | `{ stances: list[{url, chunk_text, stance, confidence}] }` |
| Research basis | RAG + LLM stance classification shown in JMIR 2025 to substantially improve accuracy and reduce hallucinations vs. LLM-only fact-checking. |

---

#### L4e — Aggregate Corroboration Score

| Item | Detail |
|---|---|
| Method | Count SUPPORTS vs REFUTES across all chunks (ignoring IRRELEVANT). `corroboration_score = supports / (supports + refutes)` if any stances found, else 0.5 (neutral) |
| Output | `{ layer: 4, checkable: true, post_type: str, corroboration_score: float, sources: list[{url, stance, chunk_preview}], error: None }` |
| File | `layers/layer4_crossref.py` |

**Post type routing summary:**

| Post Type | L4 Behavior | Verdict impact |
|---|---|---|
| Factual claim | Full L4b→L4e pipeline | corroboration_score used in final score |
| Opinion | Returns immediately | Final verdict = NOT FACT-CHECKABLE |
| Satire / humor | Returns immediately | Final verdict = SATIRE DETECTED |
| Lifestyle / personal | Returns immediately | Final verdict = SUBJECTIVE CONTENT |
| Quote attribution | Full pipeline, query focused on speaker | corroboration_score used in final score |

---

### Layer 5 — Plain-English Verdict (TinyLlama)

| Item | Detail |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (HuggingFace, free) |
| Input | Structured prompt: CLIP score, DistilBERT label, cross-reference score, BLIP-2 caption |
| Output | Plain-English paragraph verdict + final label |
| Fallback | If TinyLlama OOMs, use rule-based verdict from weighted average |
| File | `layers/layer5_verdict.py` |

**Prompt template:**
```
You are a fact-checking assistant. Given these signals:
- Post type: {post_type}
- Caption credibility (DistilBERT): {label} ({confidence:.0%})
- Caption-image consistency (CLIP): {similarity:.0%}
- Auto-generated image description (BLIP-2): "{blip_caption}"
- Fact-checkable: {checkable}
- Web evidence: {n_supporting} sources support, {n_refuting} sources refute the claim

If the post is NOT fact-checkable, explain why and label it accordingly.
Otherwise, write a 2-3 sentence verdict explaining whether this post is likely fake, real, or uncertain.
Start with one of: LIKELY FAKE / LIKELY REAL / UNCERTAIN / NOT FACT-CHECKABLE / SATIRE DETECTED / SUBJECTIVE CONTENT.
```

---

### Layer 6 — BLIP-2 Visual Captioning (v2 Upgrade)

| Item | Detail |
|---|---|
| Model | `Salesforce/blip2-opt-2.7b` (HuggingFace) |
| Purpose | Auto-generate a caption from the image → re-run CLIP between user caption and BLIP-2 caption (text-to-text) for a more semantically rich consistency check |
| Research basis | CoVLM (arXiv 2410.04426) demonstrates that CLIP+BLIP consensus pseudo-labels are far more robust than CLIP alone. ConDA-TTA (arXiv 2406.07430) uses BLIP-2 as multimodal feature encoder on NewsCLIPpings and achieves domain-invariant out-of-context detection. |
| Output | `{ blip_caption: str, text_text_similarity: float }` |
| File | `layers/layer2_clip.py` (as `blip2_verify()` function) |

**Why this matters for v2:** Pure CLIP cosine similarity is known to exploit shortcuts (WACV 2025). Adding BLIP-2's auto-caption as a second text to compare against gives a text↔text similarity that is harder to fool with superficially matching images.

---

## 3. Scoring & Verdict Logic

```python
def compute_final_verdict(l1, l2, l3, l4, l6):
    # Short-circuit: post is not fact-checkable
    if not l4.get("checkable"):
        return l4.get("post_type", "NOT FACT-CHECKABLE")

    # Weighted aggregate (only runs for checkable claims)
    text_score   = l1["confidence"] if l1["label"] == "FALSE" else 1 - l1["confidence"]
    clip_score   = 1 - l2["similarity"]           # high mismatch = high suspicion
    blip_score   = 1 - l6["text_text_similarity"]
    source_score = 1 - l4["corroboration_score"]  # low corroboration = high suspicion
    video_score  = 1 - l3["avg_similarity"] if l3 else None

    weights = [0.25, 0.20, 0.15, 0.30, 0.10]  # text, clip, blip, source, video
    # source weight raised to 0.30 — open-web RAG evidence is now richer and more reliable
    scores  = [text_score, clip_score, blip_score, source_score, video_score or 0]
    final   = sum(w * s for w, s in zip(weights, scores))

    if final > 0.65:   return "LIKELY FAKE"
    if final < 0.35:   return "LIKELY REAL"
    return "UNCERTAIN"
```

**Weight change rationale:** Source weight raised from 0.20 → 0.30 because the new RAG pipeline retrieves semantically relevant open-web evidence with per-chunk stance scoring — far more signal than the old static API lookup. BLIP score weight reduced from 0.20 → 0.15 to compensate.

---

## 4. Research Foundation

| Paper | Year | Key Takeaway for Veritas |
|---|---|---|
| Setiawan et al. — CLIP+DistilBERT fake news, PLoS ONE | 2025 | CLIP+MLP achieves F1 86% on Fakeddit; LoRA prevents overfitting |
| FND-LLM (PMC) | 2024 | SLM+LLM combo outperforms pure SLM; co-attention fusion |
| CoVLM — CLIP+BLIP consensus | 2024 | BLIP-generated descriptions improve pseudo-label robustness |
| Papadopoulos et al. (WACV) | 2025 | CLIP alone exploits shortcuts; evidence retrieval needed |
| ConDA-TTA — BLIP-2 OOC detection | 2024 | BLIP-2 feature encoder best for domain-adaptive OOC detection |
| TT-BLIP (arXiv 2403.12481) | 2024–2025 | Tri-transformer BLIP fusion achieves 94–96% on Weibo/Fakeddit |
| VMID (arXiv 2411.10032) | 2024 | LLM-based short video misinformation detection via keyframes |
| Frontiers explainable multimodal | 2025 | SHAP/LIME interpretability required for journalist adoption |
| Ensemble ViLBERT+DistilBERT (MDPI) | Jan 2026 | Ensemble with soft-voting + T5 metaclassifier: 96% Fakeddit |
| AVeriTeC dataset | 2024 | 4,568 real-world claims verified against open web; defines 5 checkable claim types; mandatory check-worthiness gate before retrieval |
| RAG fact-checking — JMIR | 2025 | RAG + LLM improves accuracy, reduces hallucination, cites sources; open-web retrieval beats static source lists |
| LLM claim verification via web search | 2025 | Verifying 17,856 claims via search API + LLM reasoning; Serper.dev used as retrieval backend |
| Automated fact-checking survey | 2024–2025 | Claim detection → evidence retrieval → verification is the mandatory 3-stage pipeline; skipping stage 1 degrades all downstream results |

---

## 5. Tech Stack

| Component | Technology |
|---|---|
| Frontend | Gradio 4.x |
| Text model | `distilbert-base-uncased` + LoRA (PEFT) |
| Vision-language | `ViT-B/32` OpenCLIP + `blip2-opt-2.7b` |
| Video | OpenCV `cv2.VideoCapture` |
| Verdict LLM | `TinyLlama-1.1B-Chat-v1.0` |
| Check-worthiness | TinyLlama zero-shot prompt |
| Web search | Serper.dev (primary) → Tavily (fallback) → DuckDuckGo (last resort) |
| Web scraping | `requests` + `beautifulsoup4` |
| Stance classification | TinyLlama per-chunk prompt |
| Training data | HuggingFace `liar` dataset |
| Runtime | Python 3.10+, PyTorch 2.2+ |
| GPU optional | CPU-safe with `torch.float32` fallback |

**Removed:** `praw` (Reddit), `wikipedia-api`, `gnews` — all replaced by open-web RAG retrieval.

---

## 6. Directory Structure

```
Veritas/
├── app.py                    # Gradio UI entry point
├── check_setup.py            # Dependency checker
├── requirements.txt          # All v2 deps
├── train.ipynb               # LIAR fine-tune notebook
├── .env.example              # API key template (SERPER_API_KEY, TAVILY_API_KEY)
├── layers/
│   ├── __init__.py
│   ├── layer1_text.py        # DistilBERT credibility
│   ├── layer2_clip.py        # CLIP + BLIP-2 consistency
│   ├── layer3_video.py       # OpenCV + CLIP video
│   ├── layer4_crossref.py    # RAG cross-ref (L4a→L4e)
│   └── layer5_verdict.py     # TinyLlama verdict
├── models/                   # Cached fine-tuned weights (gitignored)
├── tests/
│   ├── test_layer1.py
│   ├── test_layer2.py
│   ├── test_layer3.py
│   ├── test_layer4.py
│   └── test_layer5.py
├── STATE.md                  # Live project state
├── ARCHITECTURE.md           # This file
└── .github/
    └── copilot-instructions.md
```

---

## 7. Performance Targets

| Metric | Target | Basis |
|---|---|---|
| Text classification F1 | ≥ 0.78 | LIAR benchmark baseline |
| Caption-image mismatch accuracy | ≥ 0.82 | NewsCLIPpings CLIP baseline |
| Check-worthiness gate accuracy | ≥ 0.85 | AVeriTeC claim type taxonomy |
| End-to-end verdict accuracy (checkable posts) | ≥ 0.85 | Ensemble research targets |
| Inference time — CPU, image post | < 20s | Gradio UX requirement (RAG adds ~5s) |
| Inference time — GPU, image post | < 8s | Deployment target |
| Web retrieval per claim | ≤ 5 URLs scraped | Rate limit + latency budget |
