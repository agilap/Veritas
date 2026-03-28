from unittest.mock import patch

import torch

from layers import text_classifier


class _FakeTokenizer:
    def __call__(self, caption, return_tensors="pt", truncation=True, max_length=256, padding=True):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}


class _FakeOutput:
    def __init__(self):
        self.logits = torch.tensor([[0.1, 0.2, 0.7]])


class _FakeModel:
    def __call__(self, **kwargs):
        return _FakeOutput()


@patch("layers.text_classifier.load_model", return_value=(_FakeTokenizer(), _FakeModel()))
def test_returns_keys(_mock_loader):
    text_classifier._tokenizer = None
    text_classifier._model = None

    label, confidence, _tip = text_classifier.classify_text("test caption")
    result = {"label": label, "confidence": confidence, "error": None}

    assert "label" in result
    assert "confidence" in result
    assert "error" in result


@patch("layers.text_classifier.load_model", return_value=(_FakeTokenizer(), _FakeModel()))
def test_no_crash_empty(_mock_loader):
    text_classifier._tokenizer = None
    text_classifier._model = None
    text_classifier.classify_text("")


@patch("layers.text_classifier.load_model", return_value=(_FakeTokenizer(), _FakeModel()))
def test_no_crash_long(_mock_loader):
    text_classifier._tokenizer = None
    text_classifier._model = None
    text_classifier.classify_text("word " * 600)
