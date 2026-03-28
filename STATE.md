# State

Layer status:

- Layer 0: ✅ Complete
- Layer 1: ✅ Complete
- Layer 2: ✅ Partial — CLIP done, BLIP missing
- Layer 3: ✅ Complete
- Layer 4: ⚠️ Needs rewrite — old Wikipedia/GNews -> RAG pipeline
- Layer 5: ⚠️ Needs quantization — TinyLlama not 4-bit quantized
- UI: ✅ Complete

## Changelog

| Date | Change | Notes |
| --- | --- | --- |
| 2026-03-28 | Documented current layer status | Added baseline status for Layers 0-5 and UI |

## Dependencies State

| Dependency | Current State | Required State | Notes |
| --- | --- | --- | --- |
| DistilBERT | In use for Layer 1 | Keep as-is | LIAR dataset classifier |
| CLIP ViT-B/32 | In use for Layers 2 and 3 | Keep as-is | Cosine similarity checks |
| BLIP | Missing | Add to Layer 2 | Needed for blip_verify() |
| OpenCV | In use for Layer 3 | Keep as-is | Keyframe extraction |
| TinyLlama | In use without 4-bit NF4 | Use 4-bit NF4 quantization | Required for 4GB VRAM |
| Wikipedia/GNews stack | Legacy Layer 4 path | Replace with RAG pipeline | Must be removed from active flow |
