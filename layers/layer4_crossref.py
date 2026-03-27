def check_worthiness(caption: str) -> dict:
    return {
        "layer": "4a",
        "error": None,
        "checkable": None,
        "post_type": None,
        "caption": caption,
    }


def web_search(caption: str) -> dict:
    return {
        "layer": "4b",
        "error": None,
        "queries": [],
        "result_urls": [],
        "caption": caption,
    }


def scrape_evidence(caption: str) -> dict:
    return {
        "layer": "4c",
        "error": None,
        "evidence_chunks": [],
        "caption": caption,
    }


def classify_stance(caption: str) -> dict:
    return {
        "layer": "4d",
        "error": None,
        "stances": [],
        "caption": caption,
    }


def aggregate_score(caption: str) -> dict:
    return {
        "layer": "4e",
        "error": None,
        "corroboration_score": None,
        "caption": caption,
    }


def cross_reference(caption: str) -> dict:
    worthiness = check_worthiness(caption)
    search = web_search(caption)
    evidence = scrape_evidence(caption)
    stance = classify_stance(caption)
    aggregate = aggregate_score(caption)

    return {
        "layer": 4,
        "error": None,
        "caption": caption,
        "check_worthiness": worthiness,
        "web_search": search,
        "scrape_evidence": evidence,
        "classify_stance": stance,
        "aggregate_score": aggregate,
    }