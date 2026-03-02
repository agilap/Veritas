"""
Layer 4 — Multi-Source Cross-Reference
Searches the claim across Reddit (PRAW), Wikipedia, and GNews.
Returns a weighted credibility signal and a list of source snippets.

Environment variables needed (add to .env or HF Space secrets):
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT          (e.g.  "TruthScan/1.0")
    GNEWS_API_KEY              (free tier at https://gnews.io — 100 req/day)
"""

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SourceResult:
    source:  str        # "Reddit" | "Wikipedia" | "GNews"
    title:   str
    snippet: str
    url:     str
    supports_claim: bool   # True = corroborates | False = contradicts/unrelated
    weight:  float = 1.0  # credibility weight of this source type


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str, max_len: int = 200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return textwrap.shorten(text, width=max_len, placeholder="…")


def _keywords(caption: str, n: int = 6) -> str:
    """Extract the n most meaningful words (naive — no stopword lib needed)."""
    stopwords = {
        "a","an","the","is","are","was","were","be","been","being","have",
        "has","had","do","does","did","will","would","could","should","may",
        "might","must","shall","can","need","dare","ought","used","to","of",
        "in","for","on","with","at","by","from","up","about","into","through",
        "during","before","after","above","below","between","each","all","both",
        "few","more","most","other","some","such","no","nor","not","only","same",
        "so","than","too","very","s","t","just","don","and","or","but","if","that",
        "this","these","those","it","its","they","them","their","we","our","my",
        "you","your","he","she","his","her","i","me","us","what","who","which",
        "when","where","why","how","says","said","according","claim","claims",
    }
    words = re.findall(r"[a-zA-Z]{4,}", caption.lower())
    seen  = set()
    result = []
    for w in words:
        if w not in stopwords and w not in seen:
            result.append(w)
            seen.add(w)
        if len(result) == n:
            break
    return " ".join(result)


# ── Reddit ────────────────────────────────────────────────────────────────────

def _search_reddit(query: str, limit: int = 5) -> List[SourceResult]:
    try:
        import praw
        reddit = praw.Reddit(
            client_id     = os.environ["REDDIT_CLIENT_ID"],
            client_secret = os.environ["REDDIT_CLIENT_SECRET"],
            user_agent    = os.environ.get("REDDIT_USER_AGENT", "TruthScan/1.0"),
        )
        results = []
        for post in reddit.subreddit("all").search(query, limit=limit, sort="relevance"):
            # Heuristic: high upvote ratio on r/factcheck / r/news → corroborates
            supports = post.upvote_ratio >= 0.65
            results.append(SourceResult(
                source="Reddit",
                title=_clean(post.title, 120),
                snippet=_clean(post.selftext or post.title, 200),
                url=f"https://reddit.com{post.permalink}",
                supports_claim=supports,
                weight=0.6,  # Reddit is lower credibility than news
            ))
        return results
    except Exception as e:
        print(f"[Reddit] {e}")
        return []


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _search_wikipedia(query: str, sentences: int = 3) -> List[SourceResult]:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        titles   = wikipedia.search(query, results=2)
        results  = []
        for title in titles:
            try:
                page    = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                results.append(SourceResult(
                    source="Wikipedia",
                    title=page.title,
                    snippet=_clean(summary, 200),
                    url=page.url,
                    supports_claim=True,  # Wikipedia finding = claim topic exists (neutral)
                    weight=0.9,
                ))
            except Exception:
                pass
        return results
    except Exception as e:
        print(f"[Wikipedia] {e}")
        return []


# ── GNews ─────────────────────────────────────────────────────────────────────

def _search_gnews(query: str, max_results: int = 5) -> List[SourceResult]:
    try:
        import requests
        api_key = os.environ.get("GNEWS_API_KEY", "")
        if not api_key:
            return []
        url     = "https://gnews.io/api/v4/search"
        params  = {
            "q":       query,
            "lang":    "en",
            "max":     max_results,
            "apikey":  api_key,
        }
        resp    = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results  = []
        for art in articles:
            results.append(SourceResult(
                source="GNews",
                title=_clean(art.get("title", ""), 120),
                snippet=_clean(art.get("description", ""), 200),
                url=art.get("url", ""),
                supports_claim=True,   # presence in news = topic is real; LLM interprets
                weight=1.0,
            ))
        return results
    except Exception as e:
        print(f"[GNews] {e}")
        return []


# ── Main entry ────────────────────────────────────────────────────────────────

def cross_reference(caption: str) -> Tuple[float, List[SourceResult]]:
    """
    Returns:
        corroboration_score  — 0.0 (fully contradicted) to 1.0 (fully corroborated)
        sources              — list of SourceResult objects for display
    """
    keywords = _keywords(caption)
    all_results: List[SourceResult] = []

    all_results += _search_reddit(keywords)
    all_results += _search_wikipedia(keywords)
    all_results += _search_gnews(keywords)

    if not all_results:
        return 0.5, []  # neutral when no API keys / network

    total_weight    = sum(r.weight for r in all_results)
    support_weight  = sum(r.weight for r in all_results if r.supports_claim)
    score = support_weight / total_weight if total_weight > 0 else 0.5

    return round(score, 3), all_results


if __name__ == "__main__":
    score, sources = cross_reference("Earthquake strikes Turkey killing thousands")
    print(f"Corroboration score: {score:.1%}  ({len(sources)} sources found)")
    for s in sources[:3]:
        print(f"  [{s.source}] {s.title}")
        print(f"  {s.snippet[:80]}…")
        print(f"  {s.url}\n")
