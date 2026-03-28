"""
Layer 4 - Open-web RAG (staged)
Current stage: check-worthiness gate only.
"""

import os
import importlib
from functools import lru_cache
from typing import Dict, List
from urllib.parse import quote_plus

import requests
import torch
from bs4 import BeautifulSoup
from tavily import TavilyClient
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
POST_TYPES = ["FACTUAL_CLAIM", "OPINION", "SATIRE", "LIFESTYLE", "PERSONAL", "OTHER"]
STANCE_TYPES = ["SUPPORTS", "REFUTES", "IRRELEVANT"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TINYLLAMA_AVAILABLE = True
_TINYLLAMA_ERROR = None


@lru_cache(maxsize=1)
def _load_tinyllama():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


try:
    _load_tinyllama()
except Exception as exc:
    TINYLLAMA_AVAILABLE = False
    _TINYLLAMA_ERROR = str(exc)
    print(f"[Layer4] TinyLlama gate unavailable: {exc}")


def _extract_post_type(text: str) -> str:
    upper = text.upper()
    for post_type in POST_TYPES:
        if post_type in upper:
            return post_type
    return "FACTUAL_CLAIM"


def _check_worthiness(caption: str) -> Dict[str, object]:
    prompt = f"""Classify this social media post into exactly one category.
Post: "{caption}"
Categories:
- FACTUAL_CLAIM: verifiable statement about events, statistics, quotes, or facts
- OPINION: personal view or belief
- SATIRE: humor, parody, or irony
- LIFESTYLE: food, travel, fashion, fitness, personal life
- PERSONAL: personal story or experience
- OTHER: anything else
Respond with the category name only."""

    if not TINYLLAMA_AVAILABLE:
        post_type = "FACTUAL_CLAIM"
        return {"checkable": True, "post_type": post_type}

    try:
        tokenizer, model = _load_tinyllama()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        input_device = model.device if hasattr(model, "device") else torch.device(DEVICE)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(prompt):] if len(decoded) > len(prompt) else decoded
        post_type = _extract_post_type(response_text)
    except Exception:
        post_type = "FACTUAL_CLAIM"

    checkable = post_type == "FACTUAL_CLAIM"
    return {"checkable": checkable, "post_type": post_type}


def _rewrite_queries(caption: str) -> List[str]:
    default_queries = [
        f"evidence supporting claim: {caption}",
        f"evidence refuting claim: {caption}",
    ]

    if not TINYLLAMA_AVAILABLE:
        return default_queries

    prompt = f"""Rewrite this post into exactly two neutral web-search queries.
Post: "{caption}"
Query 1 should seek confirming evidence.
Query 2 should seek refuting evidence.
Output format:
QUERY_1: <text>
QUERY_2: <text>"""

    try:
        tokenizer, model = _load_tinyllama()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_device = model.device if hasattr(model, "device") else torch.device(DEVICE)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(prompt):] if len(decoded) > len(prompt) else decoded

        q1, q2 = None, None
        for line in response_text.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("QUERY_1:"):
                q1 = stripped.split(":", 1)[1].strip()
            elif upper.startswith("QUERY_2:"):
                q2 = stripped.split(":", 1)[1].strip()

        if q1 and q2:
            return [q1, q2]
    except Exception:
        pass

    return default_queries


def _search_serper(query: str) -> List[str]:
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"x-api-key": api_key},
            json={"q": query, "num": 5},
            timeout=5,
        )
        if resp.status_code != 200:
            return []

        payload = resp.json()
        urls: List[str] = []
        for item in payload.get("organic", []):
            link = str(item.get("link", "")).strip()
            if link:
                urls.append(link)
        return urls
    except Exception:
        return []


def _search_tavily(query: str) -> List[str]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        client = TavilyClient(api_key=api_key)
        payload = client.search(query=query, max_results=5)
        urls: List[str] = []
        for item in payload.get("results", []):
            link = str(item.get("url", "")).strip()
            if link:
                urls.append(link)
        return urls
    except Exception:
        return []


def _search_duckduckgo(query: str) -> List[str]:
    try:
        endpoint = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
        resp = requests.get(endpoint, timeout=5)
        if resp.status_code != 200:
            return []

        payload = resp.json()
        urls: List[str] = []

        abstract_url = str(payload.get("AbstractURL", "")).strip()
        if abstract_url:
            urls.append(abstract_url)

        for topic in payload.get("RelatedTopics", []):
            if isinstance(topic, dict) and "FirstURL" in topic:
                link = str(topic.get("FirstURL", "")).strip()
                if link:
                    urls.append(link)
            elif isinstance(topic, dict) and "Topics" in topic:
                for sub in topic.get("Topics", []):
                    link = str(sub.get("FirstURL", "")).strip()
                    if link:
                        urls.append(link)

        return urls
    except Exception:
        return []


def _split_chunks(text: str, words_per_chunk: int = 300) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _retrieve_evidence(caption: str) -> dict:
    queries = _rewrite_queries(caption)

    collected_urls: List[str] = []
    seen = set()

    for query in queries:
        urls = _search_serper(query)
        if not urls:
            urls = _search_tavily(query)
        if not urls:
            urls = _search_duckduckgo(query)

        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            collected_urls.append(url)
            if len(collected_urls) >= 8:
                break
        if len(collected_urls) >= 8:
            break

    evidence_chunks = []
    for url in collected_urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue
            text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
            if not text:
                continue

            chunks = _split_chunks(text, words_per_chunk=300)
            if not chunks:
                continue

            try:
                tfidf_module = importlib.import_module("sklearn.feature_extraction.text")
                pairwise_module = importlib.import_module("sklearn.metrics.pairwise")
                TfidfVectorizer = getattr(tfidf_module, "TfidfVectorizer")
                cosine_similarity = getattr(pairwise_module, "cosine_similarity")
            except Exception:
                continue

            tfidf = TfidfVectorizer()
            vectors = tfidf.fit_transform([caption] + chunks)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:2]
            for idx, score in ranked:
                evidence_chunks.append(
                    {
                        "url": url,
                        "text": chunks[idx],
                        "relevance": float(round(float(score), 4)),
                    }
                )
        except Exception:
            # Skip bad URLs silently.
            continue

    return {
        "queries": queries,
        "evidence_chunks": evidence_chunks,
        "error": None,
    }


def _classify_stance(caption: str, chunk_text: str) -> str:
    prompt = f"""Claim: "{caption}"
Evidence: "{chunk_text[:400]}"
Does this evidence SUPPORT, REFUTE, or is it IRRELEVANT to the claim?
Answer with one word only: SUPPORTS, REFUTES, or IRRELEVANT."""

    if not TINYLLAMA_AVAILABLE:
        return "IRRELEVANT"

    try:
        tokenizer, model = _load_tinyllama()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        input_device = model.device if hasattr(model, "device") else torch.device(DEVICE)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(prompt):] if len(decoded) > len(prompt) else decoded
        upper = response_text.upper()
        for stance in STANCE_TYPES:
            if stance in upper:
                return stance
    except Exception:
        pass

    return "IRRELEVANT"


def _aggregate_score(stances: List[str]) -> float:
    supports = sum(1 for s in stances if s == "SUPPORTS")
    refutes = sum(1 for s in stances if s == "REFUTES")

    usable = supports + refutes
    if usable == 0:
        return 0.5
    return supports / usable


def cross_reference(caption: str) -> dict:
    gate = _check_worthiness(caption)
    if not gate["checkable"]:
        return {
            "checkable": False,
            "post_type": gate["post_type"],
            "corroboration_score": None,
            "sources": [],
            "error": None,
        }

    retrieval = _retrieve_evidence(caption)
    if retrieval.get("error"):
        return {
            "checkable": True,
            "post_type": "FACTUAL_CLAIM",
            "corroboration_score": 0.5,
            "sources": [],
            "error": retrieval.get("error"),
        }

    stances: List[str] = []
    sources = []
    for chunk in retrieval.get("evidence_chunks", []):
        chunk_text = str(chunk.get("text", ""))
        stance = _classify_stance(caption, chunk_text)
        stances.append(stance)
        sources.append(
            {
                "url": str(chunk.get("url", "")),
                "stance": stance,
                "chunk_preview": chunk_text[:150],
            }
        )

    corroboration_score = float(_aggregate_score(stances))
    return {
        "checkable": True,
        "post_type": "FACTUAL_CLAIM",
        "corroboration_score": corroboration_score,
        "sources": sources,
        "error": None,
    }
