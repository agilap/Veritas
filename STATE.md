# Veritas v2 — STATE.md
> **Auto-updated by Copilot on every meaningful action.** Each entry records what changed, what is now working, and what is still pending.

---

## 🗂 Project Overview
| Field | Value |
|---|---|
| Project | Veritas v2 — TruthScan |
| Repo | https://github.com/agilap/Veritas |
| Branch | `main` |
| Last Updated | 2026-03-15 |
| Overall Status | 🟡 In Progress |

---

## ✅ Layer Completion Tracker

| Layer | Name | Status | Last Action | Notes |
|---|---|---|---|---|
| L0 | Project Scaffold | ✅ Complete | 2026-03-15 | Scaffold completed; v2 env template and setup checks updated |
| L1 | Text Classifier (DistilBERT / LIAR) | 🟡 In Progress | 2026-03-15 | Implemented `analyze_text()` with cached DistilBERT + optional LoRA adapter load |
| L2 | Caption↔Image CLIP Consistency | ⬜ Not Started | — | Zero-shot cosine similarity |
| L3 | Caption↔Video Consistency | ⬜ Not Started | — | OpenCV keyframe extraction + CLIP |
| L4 | RAG Cross-Reference (Open Web) | ⬜ Not Started | — | L4a check-worthiness gate + L4b–L4e open-web RAG. Reddit/Wikipedia/GNews removed. |
| L5 | LLM Verdict (TinyLlama) | ⬜ Not Started | — | Synthesize all signals |
| L6 | BLIP-2 Visual Captioning (v2 upgrade) | ⬜ Not Started | — | Replace naive CLIP with richer caption context |
| UI | Gradio Interface | ⬜ Not Started | — | Upload caption + image/video |

---

## 📁 File System State

```
Veritas/
├── app.py                  # ⬜ Needs v2 rewrite
├── check_setup.py          # ✅ Updated (v2 checks)
├── requirements.txt        # ✅ Updated (v2 pinned deps)
├── train.ipynb             # ⬜ Needs LIAR fine-tune update
├── .env.example            # ✅ Updated (Serper/Tavily keys only)
├── layers/
│   ├── layer1_text.py      # ✅ Created (stub)
│   ├── layer2_clip.py      # ✅ Created (stub)
│   ├── layer3_video.py     # ✅ Created (stub)
│   ├── layer4_crossref.py  # ✅ Created (stub)
│   └── layer5_verdict.py   # ✅ Created (stub)
├── tests/
│   ├── test_layer1.py      # ✅ Created (placeholder)
│   ├── test_layer2.py      # ✅ Created (placeholder)
│   ├── test_layer3.py      # ✅ Created (placeholder)
│   ├── test_layer4.py      # ✅ Created (placeholder)
│   └── test_layer5.py      # ✅ Created (placeholder)
├── STATE.md                # ✅ This file
├── ARCHITECTURE.md         # ✅ Created
└── .github/
    └── copilot-instructions.md  # ✅ Created
```

---

## 🧪 Test Results

| Test | Status | Last Run | Score |
|---|---|---|---|
| DistilBERT unit test | ⬜ Pending | — | — |
| CLIP similarity sanity | ⬜ Pending | — | — |
| Video keyframe extract | ⬜ Pending | — | — |
| GNews API ping | ⬜ Pending | — | — |
| End-to-end Gradio smoke | ⬜ Pending | — | — |

---

## 🐛 Known Issues / Blockers

_None yet. Copilot adds entries here as bugs are found and fixed._

| # | Description | Status | Fixed In |
|---|---|---|---|
| — | — | — | — |

---

## 📦 Dependencies State

| Package | Version | Status |
|---|---|---|
| transformers | ≥4.40 | ⬜ Not installed |
| torch | ≥2.2 | ⬜ Not installed |
| open-clip-torch | ≥2.24 | ⬜ Not installed |
| opencv-python | ≥4.9 | ⬜ Not installed |
| gradio | ≥4.0 | ⬜ Not installed |
| requests | ≥2.31 | ⬜ Not installed |
| beautifulsoup4 | ≥4.12 | ⬜ Not installed |
| datasets | ≥2.18 | ⬜ Not installed |
| accelerate | ≥0.29 | ⬜ Not installed |
| peft | ≥0.10 | ⬜ Not installed |
| python-dotenv | ≥1.0 | ⬜ Not installed |
| pillow | ≥10.0 | ⬜ Not installed |
| tavily-python | ≥0.3 | ⬜ Not installed |
| ~~praw~~ | ~~removed~~ | ❌ Not used |
| ~~wikipedia-api~~ | ~~removed~~ | ❌ Not used |
| ~~gnews~~ | ~~removed~~ | ❌ Not used |

---

## 🔄 Changelog

| Date | Author | Change |
|---|---|---|
| Init | Copilot | STATE.md created from project spec |
| 2026-03-15 | User | Removed Reddit/PRAW — too complex to set up |
| 2026-03-15 | User | Layer 4 redesigned: Wikipedia + GNews + Reddit → open-web RAG pipeline with check-worthiness gate (L4a→L4e). Research basis: AVeriTeC 2024, JMIR RAG 2025, automated fact-checking survey. |
| 2026-03-15 | Copilot | Updated ARCHITECTURE.md, STATE.md, copilot-instructions.md to reflect Layer 4 redesign. Scoring weights updated: source 0.20→0.30, blip 0.20→0.15. Verdict labels expanded to include NOT FACT-CHECKABLE / SATIRE DETECTED / SUBJECTIVE CONTENT. |
| 2026-03-15 | Copilot | Scaffolded v2 layer modules (`layer1_text`→`layer5_verdict`) and added placeholder tests (`test_layer1`→`test_layer5`). Marked L0 Project Scaffold as In Progress and updated file system state. |
| 2026-03-15 | Copilot | Rewrote `requirements.txt` to pinned Veritas v2 dependencies, added optional `salesforce-lavis` comment, and synced Dependencies State table (including removed `praw`/`wikipedia-api`/`gnews`). |
| 2026-03-15 | Copilot | Updated `.env.example` to only `SERPER_API_KEY` and `TAVILY_API_KEY` (with free-tier comments) and rewrote `check_setup.py` to enforce required env keys + import checks for all `requirements.txt` packages with pass/fail exit code. |
| 2026-03-15 | Copilot | Fully implemented `layers/layer1_text.py`: cached module-level DistilBERT loader, optional PEFT LoRA (`r=16`, `q_lin`/`v_lin`, `SEQ_CLS`), 3-class output (TRUE/HALF-TRUE/FALSE), 512-token truncation, and required error-safe return schema. |
| 2026-03-15 | Copilot | Rewrote `train.ipynb` with an 8-step LIAR fine-tuning workflow: dependency asserts, dataset load, 6→3 label collapse, tokenization, LoRA setup, Trainer training (`3` epochs, batch `16`, `eval_strategy="epoch"`, `f1_macro`), per-epoch F1 printout, and adapter save to `models/liar_lora/`. |

---

> **Copilot instruction:** After every file creation, bug fix, test pass/fail, or layer completion, append a row to the Changelog and update the relevant rows in the tables above. Never delete history — only add to it.
