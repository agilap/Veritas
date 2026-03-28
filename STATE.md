# State

Layer status:

- Layer 0: ✅ Complete
- Layer 1: ✅ Complete
- Layer 2: ✅ Complete
- Layer 3: ✅ Complete
- Layer 4: ⚠️ Needs rewrite — old Wikipedia/GNews -> RAG pipeline
- Layer 5: ⚠️ Needs quantization — TinyLlama not 4-bit quantized
- UI: ✅ Complete

## Changelog

| Date | Change | Notes |
| --- | --- | --- |
| 2026-03-28 | Documented current layer status | Added baseline status for Layers 0-5 and UI |
| 2026-03-28 | Updated env and setup checks | Replaced .env.example (Serper/Tavily only + optional Facebook cookies), removed GNews usage, and added bitsandbytes import validation in check_setup.py |
| 2026-03-28 | Layer 2 BLIP verification added | Added blip_verify() with Salesforce/blip-image-captioning-base, safe module-level loading, CLIP text-text similarity, and integrated output fields in check_caption_image() |
| 2026-03-28 | Layer 4 check-worthiness gate scaffolded | Replaced Wikipedia/GNews logic with TinyLlama NF4-quantized _check_worthiness() and staged cross_reference() dict stub with early return for non-checkable posts |

## Dependencies State

| Dependency | Current State | Required State | Notes |
| --- | --- | --- | --- |
| DistilBERT | In use for Layer 1 | Keep as-is | LIAR dataset classifier |
| CLIP ViT-B/32 | In use for Layers 2 and 3 | Keep as-is | Cosine similarity checks |
| BLIP | In use for Layer 2 | Keep as-is | blip_verify() integrated |
| OpenCV | In use for Layer 3 | Keep as-is | Keyframe extraction |
| TinyLlama | In use without 4-bit NF4 | Use 4-bit NF4 quantization | Required for 4GB VRAM |
| Wikipedia/GNews stack | Legacy Layer 4 path | Replace with RAG pipeline | Must be removed from active flow |
