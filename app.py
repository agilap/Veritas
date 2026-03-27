"""
Veritas — URL Credibility Checker
Paste any URL → Veritas scrapes the content + media → runs all analysis layers.

Run locally:  python app.py
Deploy:       Push to a HuggingFace Space (Gradio SDK)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from layers.url_fetcher     import fetch_post, Platform, PostData
from layers.text_classifier import classify_text
from layers.clip_checker    import check_caption_image
from layers.video_checker   import check_caption_video
from layers.source_checker  import cross_reference
from layers.verdict         import generate_verdict


# ── Icons ──────────────────────────────────────────────────────────────────────

PLATFORM_ICONS = {
    Platform.INSTAGRAM: "📸",
    Platform.FACEBOOK:  "🔵",
    Platform.UNKNOWN:   "🌐",
}


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_analysis(url: str, use_llm: bool):
    """
    1. Fetch post from URL (Layer 0)
    2. Run Layers 1-5 on the extracted content
    Yields Gradio output components progressively.
    """

    if not url or not url.strip():
        empty = "⚠️ Please paste a post URL."
        yield empty, "", "", "", "", "", None, None
        return

    # ── Layer 0: Fetch post ───────────────────────────────────────────────────
    yield "⏳ Fetching post…", "", "", "", "", "", None, None

    post: PostData = fetch_post(url)

    if post.error and not post.caption:
        err_md = f"### ❌ Could not fetch post\n\n{post.error}"
        yield err_md, "", "", "", "", "", None, None
        return

    icon      = PLATFORM_ICONS[post.platform]
    platform  = post.platform.value
    author    = post.author or "Unknown"
    timestamp = post.timestamp or "—"
    caption   = post.caption or ""

    fetch_warning = f"\n\n> ⚠️ Partial fetch: {post.error}" if post.error else ""
    caption_md = caption.replace("\n", "  \n> ") if caption else "_No caption found_"
    fetch_md = (
        f"### {icon} {platform} post by **{author}**\n"
        f"*{timestamp}*\n\n"
        f"> {caption_md}"
        f"{fetch_warning}"
    )

    if not caption:
        yield (
            fetch_md,
            "⚠️ No caption text found — text analysis skipped.",
            "⚠️ No caption — visual analysis skipped.",
            "⚠️ No caption — source check skipped.",
            "", "", None, None,
        )
        return

    yield "⏳ Analysing text credibility…", fetch_md, "", "", "", "", None, None

    # ── Layer 1: Text credibility ─────────────────────────────────────────────
    text_label, text_conf, text_tip = classify_text(caption)
    text_out = (
        f"**{text_label}** ({text_conf:.1%} confidence)\n\n"
        f"{text_tip}"
    )

    yield "⏳ Checking visual consistency…", fetch_md, text_out, "", "", "", None, None

    # ── Layer 2 / 3: Visual consistency ───────────────────────────────────────
    clip_sim    = 0.5
    clip_flag   = "—"
    clip_exp    = "No media found in this post."
    visual_out  = clip_exp
    worst_frame = None
    is_video    = False
    thumb_img   = None

    if post.video_path:
        is_video = True
        clip_sim, clip_flag, clip_exp, worst_frame, worst_ts = check_caption_video(
            caption, post.video_path
        )
        visual_out = (
            f"**{clip_flag}** (avg CLIP similarity: {clip_sim*100:.1f}%)\n\n"
            f"{clip_exp}"
        )
    elif post.image_path:
        img = Image.open(post.image_path).convert("RGB")
        thumb_img = img
        clip_sim, clip_flag, clip_exp = check_caption_image(caption, img)
        visual_out = (
            f"**{clip_flag}** (CLIP similarity: {clip_sim*100:.1f}%)\n\n"
            f"{clip_exp}"
        )
    else:
        visual_out = (
            "ℹ️ No image or video could be extracted from this post.\n\n"
            "Visual consistency check is skipped."
        )

    yield "⏳ Cross-referencing sources…", fetch_md, text_out, visual_out, "", "", thumb_img, worst_frame

    # ── Layer 4: Source cross-reference ───────────────────────────────────────
    corroboration, sources = cross_reference(caption)
    if sources:
        source_lines = [f"**Corroboration score: {corroboration:.1%}** ({len(sources)} sources)\n"]
        for s in sources[:5]:
            icon_s = "✅" if s.supports_claim else "❌"
            source_lines.append(
                f"{icon_s} [{s.source}] **{s.title}**\n{s.snippet}\n[→ View source]({s.url})"
            )
        source_out = "\n\n---\n\n".join(source_lines)
    else:
        source_out = (
            "ℹ️ No API keys configured — external source check skipped.\n\n"
            "Add `GNEWS_API_KEY` "
            "to your `.env` or HF Space Secrets."
        )

    yield "⏳ Generating verdict…", fetch_md, text_out, visual_out, source_out, "", thumb_img, worst_frame

    # ── Layer 5: TinyLlama verdict ────────────────────────────────────────────
    verdict = generate_verdict(
        caption=caption,
        text_label=text_label,
        text_conf=text_conf,
        clip_sim=clip_sim,
        clip_flag=clip_flag,
        corroboration=corroboration,
        n_sources=len(sources),
        is_video=is_video,
        use_llm=use_llm,
    )

    # ── Composite authenticity score ──────────────────────────────────────────
    t_score = (
        1.0 - text_conf if "False" in text_label
        else (0.5 if "Uncertain" in text_label else 1.0) * text_conf
    )
    score_components = [t_score, clip_sim, corroboration]
    auth_score = int(sum(score_components) / len(score_components) * 100)

    if auth_score >= 70:
        bar_colour = "🟢"
    elif auth_score >= 40:
        bar_colour = "🟡"
    else:
        bar_colour = "🔴"

    bar = bar_colour * (auth_score // 10) + "⬜" * (10 - auth_score // 10)
    score_out = (
        f"### Authenticity Score: **{auth_score}/100**\n\n"
        f"{bar}\n\n"
        f"| Signal | Value |\n"
        f"|--------|-------|\n"
        f"| 📝 Text credibility | {text_label} ({text_conf:.1%}) |\n"
        f"| 🖼️ Visual consistency | {clip_sim*100:.1f}% |\n"
        f"| 🌐 Source corroboration | {corroboration:.1%} ({len(sources)} sources) |"
    )

    yield verdict, fetch_md, text_out, visual_out, source_out, score_out, thumb_img, worst_frame


# ── UI ──────────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'Georgia', serif; }
#title { text-align: center; padding: 20px 0 8px 0; }
#subtitle { text-align: center; color: #666; margin-bottom: 24px; }
#verdict-box {
    border-left: 5px solid #e74c3c;
    padding: 20px;
    border-radius: 8px;
    background: #fff5f5;
    font-size: 1.05em;
}
#score-box {
    border-left: 5px solid #2980b9;
    padding: 20px;
    border-radius: 8px;
    background: #f0f7ff;
}
#fetch-box {
    border-left: 5px solid #27ae60;
    padding: 16px;
    border-radius: 8px;
    background: #f0fff4;
}
.layer-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-top: 6px;
    background: #fafafa;
}
footer { display: none !important; }
"""

PLATFORM_HELP = """
**Supported URLs:**
- `https://www.instagram.com/p/SHORTCODE/`
- `https://www.instagram.com/reel/SHORTCODE/`
- `https://www.facebook.com/pagename/posts/123456789`
- Any public article or web page URL
"""

with gr.Blocks(title="TruthScan") as demo:

    gr.Markdown("# 🔍 Veritas", elem_id="title")
    gr.Markdown(
        "URL Credibility Checker — paste any URL, get the truth.",
        elem_id="subtitle",
    )

    with gr.Row():
        url_in = gr.Textbox(
            label="🔗 URL",
            placeholder="https://instagram.com/p/…   facebook.com/…/posts/…   or any article URL",
            scale=5,
        )
        analyze_btn = gr.Button("🔍 Analyse", variant="primary", scale=1, min_width=120)

    with gr.Accordion("ℹ️ Supported platforms & URL formats", open=False):
        gr.Markdown(PLATFORM_HELP)

    use_llm_cb = gr.Checkbox(
        label="Use TinyLlama for verdict (richer output, ~30s slower)",
        value=True,
    )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            fetch_out = gr.Markdown(
                value="_Post preview will appear here._",
                label="📄 Scraped Post",
                elem_id="fetch-box",
            )
            with gr.Row():
                thumb_out = gr.Image(
                    label="🖼️ Post Image",
                    interactive=False,
                )
                worst_frame_out = gr.Image(
                    label="🎬 Most Suspicious Frame",
                    interactive=False,
                )

        with gr.Column(scale=1):
            verdict_out = gr.Markdown(
                value="_Verdict will appear here._",
                label="🏁 Verdict",
                elem_id="verdict-box",
            )
            score_out = gr.Markdown(
                value="",
                label="📊 Score",
                elem_id="score-box",
            )

    gr.Markdown("---\n### Layer-by-Layer Analysis")

    with gr.Tabs():
        with gr.Tab("📝 Layer 1 — Text Credibility"):
            gr.Markdown("_DistilBERT fine-tuned on LIAR — 3-class credibility (False / Uncertain / Credible)_")
            text_out = gr.Markdown(elem_classes=["layer-card"])

        with gr.Tab("🖼️ Layer 2/3 — Visual Consistency"):
            gr.Markdown("_OpenAI CLIP ViT-B/32 — caption ↔ image/video semantic matching_")
            visual_out = gr.Markdown(elem_classes=["layer-card"])

        with gr.Tab("🌐 Layer 4 — Source Cross-Reference"):
            gr.Markdown("_Wikipedia · GNews · IFCN fact-checkers — corroboration score_")
            source_out = gr.Markdown(elem_classes=["layer-card"])

    gr.Markdown(
        "---\n"
        "**Veritas** · CLIP · DistilBERT · TinyLlama · "
        "[GitHub](https://github.com/your-handle/veritas) · "
        "⚠️ *For research use — always verify with primary sources.*"
    )

    OUTPUTS = [
        verdict_out, fetch_out, text_out, visual_out,
        source_out, score_out, thumb_out, worst_frame_out,
    ]

    analyze_btn.click(fn=run_analysis, inputs=[url_in, use_llm_cb], outputs=OUTPUTS)
    url_in.submit(fn=run_analysis, inputs=[url_in, use_llm_cb], outputs=OUTPUTS)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        css=CSS,
    )
