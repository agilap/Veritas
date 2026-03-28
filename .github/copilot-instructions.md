# Copilot Instructions

- Always read ARCHITECTURE.md and STATE.md before making changes
- File naming: use existing names (text_classifier.py, clip_checker.py, etc.) — do NOT rename to layer1_text.py etc.
- Update STATE.md after every change
- Do NOT use praw, gnews, or wikipedia-api
- TinyLlama must use 4-bit NF4 quantization (BitsAndBytesConfig) and be loaded once at module level
- Layer 4 source_checker.py must implement the check-worthiness gate BEFORE any web search call
