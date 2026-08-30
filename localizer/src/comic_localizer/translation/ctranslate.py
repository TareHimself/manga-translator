import os
import asyncio

import ctranslate2
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from comic_localizer.core.plugin import (
    LanguageStringArgument,
    PytorchDevicePluginArgument,
    Translator,
    TranslatorResult,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
)
from comic_localizer.utils import (
    get_default_language,
    get_default_torch_device,
    standardize_language_code,
)


class CTranslateTranslator(Translator):
    """Translates using CTranslate2-converted models for fast CPU/GPU inference"""

    def __init__(
        self,
        model_url: str = "gaudi/opus-mt-ja-en-ctranslate2",
        input_language: str = "ja",
        output_language: str = get_default_language(),
        device: torch.device = get_default_torch_device(),
    ) -> None:
        super().__init__()
        model_dir = (
            model_url if os.path.isdir(model_url) else snapshot_download(model_url)
        )
        # ctranslate2 only supports cpu/cuda, unlike the pytorch device picker
        # which can also offer mps.
        ct2_device = "cuda" if device.type == "cuda" else "cpu"
        self.translator = ctranslate2.Translator(model_dir, device=ct2_device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.input_language = input_language
        self.output_language = standardize_language_code(output_language)
        # src_lang/target prefix are only meaningful for multilingual tokenizers
        # (mBART/M2M100/NLLB/etc.) that define src_lang. MarianMT (the default
        # model) doesn't, so only apply them when the tokenizer supports it.
        self._supports_lang_pair = hasattr(self.tokenizer, "src_lang")
        self._target_prefix_token = None
        if self._supports_lang_pair:
            self.tokenizer.src_lang = self.input_language
            self._target_prefix_token = getattr(
                self.tokenizer, "lang_code_to_token", {}
            ).get(self.output_language, self.output_language)

    def predict(self, batch: list[OcrResult]):
        sources = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(x.text))
            for x in batch
        ]

        kwargs = (
            {"target_prefix": [[self._target_prefix_token] for _ in sources]}
            if self._supports_lang_pair
            else {}
        )
        results = self.translator.translate_batch(sources, **kwargs)

        texts = []
        for result in results:
            tokens = result.hypotheses[0]
            if self._supports_lang_pair and tokens[:1] == [self._target_prefix_token]:
                tokens = tokens[1:]
            texts.append(
                self.tokenizer.decode(
                    self.tokenizer.convert_tokens_to_ids(tokens),
                    skip_special_tokens=True,
                )
            )
        return texts

    async def translate(self, batch: list[OcrResult]):
        results = await asyncio.to_thread(self.predict, batch)

        return [
            TranslatorResult(text, language=self.output_language) for text in results
        ]

    @staticmethod
    def get_name() -> str:
        return "CTranslate2"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:

        return [
            StringPluginArgument(
                id="model_url",
                name="Model",
                description="The Hugging Face repo id (or local path) of a CTranslate2-converted translation model",
                default="gaudi/opus-mt-ja-en-ctranslate2",
            ),
            LanguageStringArgument(
                id="input_language",
                name="Input Language",
                description="The language to translate from",
                default="ja",
            ),
            LanguageStringArgument(
                id="output_language",
                name="Output Language",
                description="The language to translate to",
            ),
            PytorchDevicePluginArgument("device", "Device"),
        ]
