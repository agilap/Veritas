# Architecture

Current architecture of the repository as it exists:

- Layer 0: url_fetcher.py — fetch_post() scrapes Instagram/Facebook, returns PostData with caption + media
- Layer 1: text_classifier.py — classify_text(caption) -> DistilBERT on LIAR dataset
- Layer 2: clip_checker.py — check_caption_image(caption, image) -> CLIP ViT-B/32 cosine similarity. NEEDS: blip_verify() to be added
- Layer 3: video_checker.py — check_caption_video(caption, video_path) -> OpenCV keyframes + CLIP
- Layer 4: source_checker.py — cross_reference(caption) -> CURRENTLY old Wikipedia + GNews. NEEDS: full RAG rewrite (check-worthiness gate -> Serper/Tavily/DDG -> scrape -> TinyLlama stance -> aggregate)
- Layer 5: verdict.py — generate_verdict() -> TinyLlama. NEEDS: 4-bit NF4 quantization for 4GB VRAM
- app.py — Gradio Blocks UI, URL input, streams results per layer
