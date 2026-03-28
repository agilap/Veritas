from unittest.mock import patch

import numpy as np
from PIL import Image

from layers import clip_checker


@patch("layers.clip_checker.blip_verify", return_value={"blip_caption": "a test image", "text_text_similarity": 0.6, "error": None})
@patch("layers.clip_checker.embed_image", return_value=np.array([1.0, 0.0], dtype=float))
@patch("layers.clip_checker.embed_text", return_value=np.array([1.0, 0.0], dtype=float))
def test_clip_returns_keys(_mock_text, _mock_image, _mock_blip):
    result = clip_checker.check_caption_image("test", Image.new("RGB", (224, 224)))
    assert "similarity" in result
    assert "flag" in result
    assert "error" in result


@patch("layers.clip_checker.BLIP_AVAILABLE", False)
def test_blip_fallback():
    result = clip_checker.blip_verify(Image.new("RGB", (224, 224)))
    assert "blip_caption" in result
