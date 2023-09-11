from transformers import pipeline
from translator.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    PluginTextArgument,
    PluginArgument,
)


class HuggingFace(Translator):
    """Translates using hugging face models"""

    def __init__(self, model_url: str = "Helsinki-NLP/opus-mt-ja-en") -> None:
        super().__init__()
        self.pipeline = pipeline("translation", model=model_url)

    async def translate(self, ocr_result: OcrResult):
        if len(ocr_result.text.strip()) == 0:
            return TranslatorResult()

        return TranslatorResult(self.pipeline(ocr_result.text)[0]["translation_text"])

    @staticmethod
    def get_name() -> str:
        return "Hugging Face"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            PluginTextArgument(
                id="model_url",
                name="Model",
                description="The Hugging Face translation model to use",
                default="staka/fugumt-ja-en",
            )
        ]
