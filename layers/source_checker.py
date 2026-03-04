"""
Layer 4 — Multi-Source Cross-Reference (PH-prioritised)
Searches the claim across PH fact-checkers (Vera Files, Rappler),
Wikipedia, and GNews — with higher credibility weights for Philippine sources.
Returns a weighted credibility signal and a list of source snippets.

Environment variables needed (add to .env or HF Space secrets):
    GNEWS_API_KEY              (free tier at https://gnews.io — 100 req/day)
"""

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SourceResult:
    source:  str        # "VeraFiles" | "Rappler" | "Wikipedia" | "GNews"
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


# ── PH Fact-Checkers (Vera Files + Rappler) ─────────────────────────────────

def _search_ph_factcheckers(query: str, max_results: int = 5) -> List[SourceResult]:
    """
    Search PH fact-check sites via Google site-search scraping.
    Falls back gracefully if blocked.
    Highest credibility weight (1.2) — these are IFCN-certified fact-checkers.
    """
    results = []
    sites = [
        ("verafiles.org", "VeraFiles"),
        ("rappler.com/newsbreak/fact-check", "Rappler"),
    ]
    headers = {"User-Agent": "Mozilla/5.0 (TruthScan/1.0)"}

    for site_domain, source_name in sites:
        try:
            import requests as req
            from bs4 import BeautifulSoup
            search_url = f"https://www.google.com/search?q=site:{site_domain}+{query}&num={max_results}"
            resp = req.get(search_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for g in soup.select("div.g, div[data-sokoban-container]"):
                link_tag = g.find("a")
                title_tag = g.find("h3")
                snippet_tag = g.find("div", class_="VwiC3b") or g.find("span", class_="st")
                if not link_tag or not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                url = link_tag.get("href", "")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                # PH fact-check articles are about debunked claims → contradicts
                title_lower = title.lower()
                supports = not any(w in title_lower for w in [
                    "false", "fake", "misleading", "not true",
                    "fabricated", "hindi totoo", "peke", "claim",
                ])

                results.append(SourceResult(
                    source=source_name,
                    title=_clean(title, 120),
                    snippet=_clean(snippet, 200),
                    url=url,
                    supports_claim=supports,
                    weight=1.2,  # PH fact-checkers are IFCN-certified → highest weight
                ))
                if len(results) >= max_results:
                    break
        except Exception as e:
            print(f"[{source_name}] {e}")
    return results


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

    all_results += _search_ph_factcheckers(keywords)
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
