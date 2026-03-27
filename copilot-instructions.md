# Veritas v2 — Copilot Instructions
> Place this file at `.github/copilot-instructions.md`. GitHub Copilot reads it automatically for context on every task in this repo.

---

## 🤖 What This Project Is

**Veritas v2 (TruthScan)** is a multimodal fake post detector. It accepts a social media caption + an image or video, then checks:
1. Is the text itself credible? (DistilBERT fine-tuned on LIAR)
2. Does the caption match the image? (CLIP + BLIP-2)
3. Does the caption match the video? (OpenCV keyframes + CLIP)
4. Is the post even fact-checkable? If yes — does open-web evidence support or refute it? (RAG: check-worthiness gate → Serper/Tavily search → scrape → TinyLlama stance)
5. What is the plain-English verdict? (TinyLlama)

**Reddit, Wikipedia, and GNews are NOT used.** Layer 4 uses open-web RAG retrieval only.

Read `ARCHITECTURE.md` for full design. Always read `STATE.md` before starting any task.

---

## 📋 STATE.md Update Protocol

**After every action you take, you MUST update `STATE.md`.** This is non-negotiable.

### What counts as an action requiring a STATE.md update:
- Creating or modifying any `.py` or `.ipynb` file
- Installing or changing a dependency in `requirements.txt`
- Fixing a bug
- Running a test (pass or fail)
- Completing a layer
- Discovering a new issue or blocker

### How to update STATE.md:

1. **Changelog:** Append a new row to the Changelog table at the bottom:
   ```
   | {today's date} | Copilot | {one-line description of what you did} |
   ```

2. **Layer Tracker:** Update the relevant row's `Status` and `Last Action` columns:
   - ⬜ Not Started → 🔵 In Progress → 🟢 Complete
   - If broken: 🔴 Broken (note the issue)

3. **File System State:** Update the emoji next to each file:
   - ⬜ Not created → ✅ Created/Working → 🔴 Broken → 🔧 Being Fixed

4. **Test Results:** Update the test table row after any test run.

5. **Known Issues:** If you find a bug, add it to the Known Issues table immediately. When you fix it, update its Status to ✅ Fixed and add the commit/file in "Fixed In".

6. **Dependencies State:** Mark ✅ Installed for any package you add to requirements.txt.

### Example STATE.md update block:
```markdown
| 2026-03-15 | Copilot | Created layers/layer1_text.py with DistilBERT LoRA fine-tune scaffold |
```

---

## 🏗 Architecture Rules

### Layer isolation
- Each layer lives in its own file: `layers/layerN_name.py`
- Each layer file must expose exactly one primary function:
  - `layer1_text.py` → `analyze_text(caption: str) -> dict`
  - `layer2_clip.py` → `analyze_image(caption: str, image: PIL.Image) -> dict`
  - `layer3_video.py` → `analyze_video(caption: str, video_path: str) -> dict`
  - `layer4_crossref.py` → `cross_reference(caption: str) -> dict`
  - Must run sub-steps in order: L4a (check-worthiness) → if checkable: L4b (search) → L4c (scrape) → L4d (stance) → L4e (aggregate)
  - If L4a returns `checkable=False`, return immediately — do NOT call search APIs
  - If any sub-step fails, set error key and continue with neutral score (0.5)
  - `layer5_verdict.py` → `generate_verdict(l1, l2, l3, l4, l6) -> dict`
- All return dicts must include a `"layer"` key and an `"error"` key (None if no error)

### Model loading
- Use `@lru_cache(maxsize=1)` or module-level singletons for all model loads — models must load once and be reused across calls
- Always include a CPU fallback: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- BLIP-2 is large — wrap its load in a try/except and fall back to CLIP-only if OOM

### Error handling
- Every layer function must catch its own exceptions and return `{"layer": N, "error": str(e), ...}` rather than crashing the pipeline
- `app.py` must gracefully display per-layer errors in the UI without stopping the whole analysis

### Gradio UI
- Use `gr.Blocks()` layout, not `gr.Interface()`
- Show a progress bar or status text while each layer runs
- Display the worst suspicious video frame if Layer 3 is triggered
- Show a colored verdict banner: 🔴 LIKELY FAKE / 🟢 LIKELY REAL / 🟡 UNCERTAIN

---

## 🧪 Testing Rules

- Every layer function must have a corresponding test in `tests/test_layerN.py`
- Tests must be runnable with `pytest tests/` from the repo root
- Use mock/stub for external APIs (GNews, Reddit) in unit tests — never hit live APIs in tests
- After writing or editing a test, run it and record the result in STATE.md

---

## 📦 Dependency Rules

- Pin all versions in `requirements.txt` with `==` for reproducibility
- After any requirements change, update the Dependencies State table in STATE.md
- Do not add any package that requires building from source without noting it in STATE.md
- Keep BLIP-2 optional: add `# optional: requires ~6GB VRAM` comment next to it in requirements.txt

---

## 🔬 Research Alignment

When implementing each layer, you MUST align with these papers:

| Layer | Primary Research Reference |
|---|---|
| L1 DistilBERT | Setiawan et al., PLoS ONE 2025 — use LoRA rank=16 |
| L2 CLIP | NewsCLIPpings (Luo et al., EMNLP 2021) + Papadopoulos WACV 2025 |
| L2 BLIP-2 | CoVLM (arXiv 2410.04426) + ConDA-TTA (arXiv 2406.07430) |
| L3 Video | VMID (arXiv 2411.10032) — keyframe-level analysis |
| L4a Check-worthiness | AVeriTeC 2024 — 5 checkable claim types; claim detection is mandatory stage 1 |
| L4b–L4e RAG | JMIR RAG fact-checking 2025 + LLM web search verification 2025 (17,856 claims) |
| L5 Verdict | TT-BLIP tri-transformer pattern (arXiv 2403.12481) |

Do not deviate from the architecture without updating `ARCHITECTURE.md` first.

---

## 🚫 What NOT to Do

- Do NOT use Reddit, PRAW, Wikipedia API, or GNews — these are removed from the project
- Do NOT call web search APIs if L4a returns `checkable=False` — this wastes API quota
- Do NOT hardcode API keys — use `.env` + `python-dotenv`
- Do NOT load models inside request handlers — always use module-level or cached loaders
- Do NOT use `gr.Interface()` — use `gr.Blocks()`
- Do NOT skip updating `STATE.md` after any change
- Do NOT add new dependencies without updating both `requirements.txt` AND the Dependencies State table in `STATE.md`
- Do NOT return raw exceptions to the user — always display a human-readable fallback
- Do NOT produce a verdict of LIKELY FAKE / LIKELY REAL for posts flagged as NOT FACT-CHECKABLE — use the special verdict labels instead

---

## 🗂 Quick Reference: Key Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI, pipeline orchestration |
| `layers/layer1_text.py` | DistilBERT text credibility |
| `layers/layer2_clip.py` | CLIP + BLIP-2 image/caption consistency |
| `layers/layer3_video.py` | OpenCV keyframe + CLIP video consistency |
| `layers/layer4_crossref.py` | RAG cross-ref: check-worthiness gate → open-web search → scrape → stance → score |
| `layers/layer5_verdict.py` | TinyLlama plain-English verdict (handles all 6 verdict labels) |
| `requirements.txt` | All pinned dependencies (no praw, wikipedia-api, gnews) |
| `train.ipynb` | LIAR dataset fine-tuning notebook |
| `.env.example` | SERPER_API_KEY, TAVILY_API_KEY only |
| `STATE.md` | 🔴 Always update this after any change |
| `ARCHITECTURE.md` | System design — read before implementing |
