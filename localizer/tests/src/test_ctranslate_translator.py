"""Tests for CTranslateTranslator src_lang/target_prefix gating logic.

These tests monkeypatch ctranslate2.Translator, AutoTokenizer and
snapshot_download entirely so they don't download or load a real model.
"""

from comic_localizer.translation import ctranslate as ctranslate_module
from comic_localizer.core.plugin import OcrResult


class _FakeHypothesis:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _FakeCt2Translator:
    def __init__(self, model_dir, device):
        self.model_dir = model_dir
        self.device = device
        self.calls = []

    def translate_batch(self, sources, **kwargs):
        self.calls.append(kwargs)
        prefix = kwargs.get("target_prefix")
        return [
            _FakeHypothesis((prefix[i] if prefix else []) + ["out"])
            for i, _ in enumerate(sources)
        ]


class _FakeTokenizerWithoutLangPair:
    def encode(self, text):
        return [ord(c) for c in text]

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [0 for _ in tokens]

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _FakeTokenizerWithLangPair(_FakeTokenizerWithoutLangPair):
    src_lang = None
    lang_code_to_token = {"en": ">>en<<"}


def _make_translator(monkeypatch, tokenizer):
    fake_translator_holder = {}

    def fake_ct2_translator(model_dir, device):
        instance = _FakeCt2Translator(model_dir, device)
        fake_translator_holder["instance"] = instance
        return instance

    monkeypatch.setattr(ctranslate_module.ctranslate2, "Translator", fake_ct2_translator)
    monkeypatch.setattr(ctranslate_module, "snapshot_download", lambda repo_id: "/fake/dir")
    monkeypatch.setattr(
        ctranslate_module.AutoTokenizer, "from_pretrained", lambda model_dir: tokenizer
    )

    translator = ctranslate_module.CTranslateTranslator(
        model_url="fake-model", input_language="ja", output_language="en"
    )
    return translator, fake_translator_holder["instance"]


def test_predict_omits_target_prefix_for_marian_style_tokenizer(monkeypatch):
    translator, fake_ct2 = _make_translator(monkeypatch, _FakeTokenizerWithoutLangPair())

    results = translator.predict([OcrResult("hello", "ja")])

    assert fake_ct2.calls == [{}]
    assert results == ["decoded"]


def test_predict_includes_target_prefix_for_multilingual_tokenizer(monkeypatch):
    translator, fake_ct2 = _make_translator(monkeypatch, _FakeTokenizerWithLangPair())

    translator.predict([OcrResult("hello", "ja")])

    assert fake_ct2.calls == [{"target_prefix": [[">>en<<"]]}]
