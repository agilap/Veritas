from unittest.mock import patch

from layers import source_checker


@patch("layers.source_checker._retrieve_evidence")
@patch("layers.source_checker._check_worthiness", return_value={"checkable": False, "post_type": "OPINION"})
def test_opinion_short_circuits(_mock_gate, mock_retrieve):
    result = source_checker.cross_reference("I think this is good")
    assert result["checkable"] is False
    assert result["sources"] == []
    mock_retrieve.assert_not_called()


@patch("layers.source_checker._classify_stance", side_effect=["SUPPORTS", "SUPPORTS"])
@patch(
    "layers.source_checker._retrieve_evidence",
    return_value={
        "queries": ["q1", "q2"],
        "evidence_chunks": [
            {"url": "https://a.com", "text": "chunk one", "relevance": 0.8},
            {"url": "https://b.com", "text": "chunk two", "relevance": 0.7},
        ],
        "error": None,
    },
)
@patch("layers.source_checker._check_worthiness", return_value={"checkable": True, "post_type": "FACTUAL_CLAIM"})
def test_factual_runs_retrieval(_mock_gate, _mock_retrieve, _mock_stance):
    result = source_checker.cross_reference("A factual claim")
    assert result["checkable"] is True
    assert result["corroboration_score"] > 0.5


@patch("layers.source_checker._search_duckduckgo", side_effect=RuntimeError("ddg down"))
@patch("layers.source_checker._search_tavily", side_effect=RuntimeError("tavily down"))
@patch("layers.source_checker._search_serper", side_effect=RuntimeError("serper down"))
@patch("layers.source_checker._rewrite_queries", return_value=["query one", "query two"])
def test_all_apis_fail(_mock_queries, _mock_serper, _mock_tavily, _mock_ddg):
    result = source_checker._retrieve_evidence("claim text")
    assert result["error"] is not None
