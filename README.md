---
title: TruthScan
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
short_description: Social Media Fake Post Detector — paste a URL, get the truth
---

# 🔍 TruthScan — Social Media Fake Post Detector

> Paste a social media post URL → TruthScan scrapes the caption + media → checks if the caption matches the visual content AND if the text itself is credible.

---

## Supported Platforms

| Platform | URL Format | API Needed? |
|----------|-----------|-------------|
| Instagram | `instagram.com/p/SHORTCODE/` or `instagram.com/reel/SHORTCODE/` | ❌ None (public posts) |
| Facebook | `facebook.com/page/posts/ID` | ❌ None (public posts) |

---

## Architecture

```
Post URL
   │
   ▼
Layer 0: URL Fetcher
   ├─ detect_platform()     — Instagram / Facebook
   ├─ scrape post           — caption text + media URLs
   └─ download_media()      — save image/video to temp file
   │
   ├─► Layer 1: DistilBERT      — text credibility (LIAR dataset)
   ├─► Layer 2: CLIP ViT-B/32   — caption ↔ image cosine similarity
   ├─► Layer 3: OpenCV + CLIP   — caption ↔ video (keyframe analysis)
   ├─► Layer 4: Wiki + GNews    — source cross-reference
   └─► Layer 5: TinyLlama-1.1B  — plain-English verdict
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/agilap/truthscan
cd truthscan
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Then edit .env — add your GNews API key for Layer 4
```

**How to get each key:**

**GNews API key (Layer 4 — optional):**
1. Sign up at https://gnews.io (free tier = 100 req/day)
2. Copy your API key from the dashboard

**Facebook cookies (optional — only for non-public posts):**
1. Install "EditThisCookie" browser extension
2. Log into Facebook → export cookies as JSON
3. Set `FACEBOOK_COOKIES_FILE=/path/to/cookies.json` in .env

### 3. Train Layer 1 — one-time, ~10 min on a free Colab T4

Open `train.ipynb` in Google Colab and run all cells. This downloads the LIAR dataset from HuggingFace, fine-tunes DistilBERT, and saves the checkpoint to `./models/liar_distilbert/`.

> **Skip training?** App still works — Layer 1 scores are random until trained, but Layers 0, 2–5 are fully functional.

### 4. Run

```bash
python app.py
# Opens at http://localhost:7860
```

---

## Deploy to HuggingFace Spaces

```bash
# 1. Create a Gradio Space at https://huggingface.co/spaces
# 2. Add these Space Secrets:
#    GNEWS_API_KEY

git remote add space https://huggingface.co/spaces/agilap/truthscan
git push space main
```

To include the trained model:
```bash
git lfs install
git lfs track "models/**"
git add models/ .gitattributes
git commit -m "add trained DistilBERT"
git push space main
```

---

## File Structure

```
truthscan/
├── app.py                      # Gradio UI — URL input, streaming results
├── train.ipynb                 # DistilBERT training notebook (for Google Colab)
├── requirements.txt
├── .env.example
├── url_fetcher.py              # Layer 0: URL → PostData (caption + media)
├── text_classifier.py          # Layer 1: DistilBERT on LIAR dataset
├── clip_checker.py             # Layer 2: CLIP caption ↔ image
├── video_checker.py            # Layer 3: CLIP caption ↔ video keyframes
├── source_checker.py           # Layer 4: Wikipedia / GNews
└── verdict.py                  # Layer 5: TinyLlama plain-English verdict
```

---

## Known Limitations

- **Instagram:** Instaloader may be throttled on heavily scraped IPs; works well on HF Spaces
- **Facebook:** Public posts scrape cleanly; friends-only or private posts require cookie auth
- **CLIP thresholds:** Calibrated on MS-COCO — may need tuning for social media imagery
- **TinyLlama:** Small model, occasional hallucinations — treat verdict as a signal, not ground truth

---

## License

MIT — use freely, credit appreciated.
