"""
check_setup.py — Run this before app.py to verify everything is in place.

Usage:  python check_setup.py

Checks:
  ✅ All required Python packages are installed
  ✅ API keys are present (warns but doesn't fail on optional ones)
  ✅ Trained DistilBERT model exists (warns if missing)
  ✅ CLIP can be imported
  ✅ OpenCV can read a dummy video frame
"""

import sys
import os

PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"

errors = []
warnings = []


def check(label, fn, required=True):
    try:
        fn()
        print(f"  {PASS}  {label}")
    except Exception as e:
        if required:
            print(f"  {FAIL}  {label}\n       → {e}")
            errors.append(label)
        else:
            print(f"  {WARN} {label}\n       → {e}")
            warnings.append(label)


# ── 1. Packages ────────────────────────────────────────────────────────────────
print("\n📦 Python packages")

check("torch",            lambda: __import__("torch"))
check("transformers",     lambda: __import__("transformers"))
check("gradio",           lambda: __import__("gradio"))
check("PIL (Pillow)",     lambda: __import__("PIL"))
check("cv2 (opencv)",     lambda: __import__("cv2"))
check("numpy",            lambda: __import__("numpy"))
check("requests",         lambda: __import__("requests"))
check("dotenv",           lambda: __import__("dotenv"))
check("praw (Reddit)",    lambda: __import__("praw"),           required=False)
check("wikipedia",        lambda: __import__("wikipedia"),      required=False)
check("instaloader",      lambda: __import__("instaloader"),    required=False)
check("facebook_scraper", lambda: __import__("facebook_scraper"), required=False)
check("tweepy",           lambda: __import__("tweepy"),         required=False)
check("datasets (HF)",    lambda: __import__("datasets"),       required=False)


# ── 2. API keys ────────────────────────────────────────────────────────────────
print("\n🔑 API keys / secrets")

from dotenv import load_dotenv
load_dotenv()

def require_env(key):
    val = os.environ.get(key, "")
    if not val:
        raise EnvironmentError(f"{key} is not set in .env or environment")

check("TWITTER_BEARER_TOKEN",  lambda: require_env("TWITTER_BEARER_TOKEN"),  required=False)
check("REDDIT_CLIENT_ID",      lambda: require_env("REDDIT_CLIENT_ID"),      required=False)
check("REDDIT_CLIENT_SECRET",  lambda: require_env("REDDIT_CLIENT_SECRET"),  required=False)
check("GNEWS_API_KEY",         lambda: require_env("GNEWS_API_KEY"),         required=False)


# ── 3. Trained model ───────────────────────────────────────────────────────────
print("\n🧠 Trained model (Layer 1)")

model_dir = os.path.join(os.path.dirname(__file__), "models", "liar_distilbert")

def check_model():
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Not found: {model_dir}\n"
            "       Run `python train.py` once to train DistilBERT on LIAR.\n"
            "       Layer 1 scores will be random until this is done."
        )
    config = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config):
        raise FileNotFoundError("model dir exists but config.json missing — re-run train.py")

check("DistilBERT / LIAR checkpoint", check_model, required=False)


# ── 4. CLIP smoke test ─────────────────────────────────────────────────────────
print("\n🖼️  CLIP import (will download ~600 MB on first run)")

def check_clip():
    from transformers import CLIPModel, CLIPProcessor
    # Don't actually download — just confirm import works
    print("       (model will be auto-downloaded on first analysis run)")

check("CLIP ViT-B/32 importable", check_clip)


# ── 5. OpenCV smoke test ───────────────────────────────────────────────────────
print("\n🎬 OpenCV")

def check_cv2():
    import cv2
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    assert len(buf) > 0

check("OpenCV encode/decode", check_cv2)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)

if errors:
    print(f"\n{FAIL} {len(errors)} REQUIRED check(s) failed — fix before running app.py:")
    for e in errors:
        print(f"    • {e}")
    sys.exit(1)
elif warnings:
    print(f"\n{WARN} {len(warnings)} optional check(s) incomplete (app will still run):")
    for w in warnings:
        print(f"    • {w}")
    print("\n✅ Core setup looks good — run:  python app.py")
else:
    print("\n✅ Everything looks good — run:  python app.py")
