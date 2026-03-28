# State

Layer status:

- Layer 0: ✅ Complete
- Layer 1: ✅ Complete
- Layer 2: ✅ Complete
- Layer 3: ✅ Complete
- Layer 4: ✅ Complete
- Layer 5: ✅ Complete
- UI: ✅ Complete

## Changelog

| Date | Change | Notes |
| --- | --- | --- |
| 2026-03-28 | Documented current layer status | Added baseline status for Layers 0-5 and UI |
| 2026-03-28 | Updated env and setup checks | Replaced .env.example (Serper/Tavily only + optional Facebook cookies), removed GNews usage, and added bitsandbytes import validation in check_setup.py |
| 2026-03-28 | Layer 2 BLIP verification added | Added blip_verify() with Salesforce/blip-image-captioning-base, safe module-level loading, CLIP text-text similarity, and integrated output fields in check_caption_image() |
| 2026-03-28 | Layer 4 check-worthiness gate scaffolded | Replaced Wikipedia/GNews logic with TinyLlama NF4-quantized _check_worthiness() and staged cross_reference() dict stub with early return for non-checkable posts |
| 2026-03-28 | Layer 4 evidence retrieval implemented | Added _retrieve_evidence(): TinyLlama query rewrite, Serper->Tavily->DuckDuckGo fallback search, URL scraping, 300-word chunking, and TF-IDF top-chunk relevance scoring |
| 2026-03-28 | Layer 4 stance + aggregation completed | Added _classify_stance(), _aggregate_score(), and full cross_reference() orchestration with non-checkable short-circuit and corroboration output from SUPPORTS/REFUTES evidence |
| 2026-03-28 | Layer 5 TinyLlama reuse + NF4 finalized | Updated verdict.py to reuse source_checker TinyLlama loader (single app-wide instance), handle Layer 4 checkable/post_type outputs, apply new weighted formula, and fall back to rule-based verdict on model/OOM errors |

## Dependencies State

| Dependency | Current State | Required State | Notes |
| --- | --- | --- | --- |
| DistilBERT | In use for Layer 1 | Keep as-is | LIAR dataset classifier |
| CLIP ViT-B/32 | In use for Layers 2 and 3 | Keep as-is | Cosine similarity checks |
| BLIP | In use for Layer 2 | Keep as-is | blip_verify() integrated |
| OpenCV | In use for Layer 3 | Keep as-is | Keyframe extraction |
| TinyLlama | In use with shared 4-bit NF4 loader | Keep as-is | Single loader reused across Layer 4 + Layer 5 to fit 4GB VRAM |
| Wikipedia/GNews stack | Removed from active flow | Keep removed | Replaced by Serper/Tavily/DuckDuckGo + TinyLlama RAG pipeline |
