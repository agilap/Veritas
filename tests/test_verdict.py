from unittest.mock import patch

from layers import verdict


@patch("layers.verdict._VERDICT_MODEL", None)
@patch("layers.verdict._VERDICT_TOKENIZER", None)
def test_likely_fake():
    result = verdict.generate_verdict(
        caption="Claim",
        text_label="❌ Likely False",
        text_conf=0.95,
        clip_sim=0.05,
        clip_flag="❌",
        corroboration=0.05,
        n_sources=2,
        is_video=False,
        use_llm=True,
        l4={"checkable": True, "post_type": "FACTUAL_CLAIM"},
        blip_sim=0.05,
        video_sim=0.05,
    )
    assert "FAKE" in result.upper()


@patch("layers.verdict._VERDICT_MODEL", None)
@patch("layers.verdict._VERDICT_TOKENIZER", None)
def test_opinion_verdict():
    result = verdict.generate_verdict(
        caption="Opinion",
        text_label="⚠️ Uncertain",
        text_conf=0.5,
        clip_sim=0.5,
        clip_flag="⚠️",
        corroboration=0.5,
        n_sources=0,
        is_video=False,
        use_llm=True,
        l4={"checkable": False, "post_type": "OPINION"},
    )
    assert "NOT FACT-CHECKABLE" in result.upper()


@patch("layers.verdict._VERDICT_MODEL", None)
@patch("layers.verdict._VERDICT_TOKENIZER", None)
def test_no_crash_blip_unavailable():
    l6 = {"text_text_similarity": None, "error": "BLIP unavailable"}
    verdict.generate_verdict(
        caption="Claim",
        text_label="⚠️ Uncertain",
        text_conf=0.4,
        clip_sim=0.3,
        clip_flag="⚠️",
        corroboration=0.5,
        n_sources=1,
        is_video=False,
        use_llm=True,
        l4={"checkable": True, "post_type": "FACTUAL_CLAIM"},
        blip_sim=l6["text_text_similarity"],
    )
