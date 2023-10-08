import torch
from transformers import pipeline
from translator.utils import get_torch_device
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
        print("Using model",model_url)
        self.pipeline = pipeline("translation", model=model_url, device=get_torch_device())

        # if torch.cuda.is_available():
        #     self.pipeline.cuda()
        # elif torch.backends.mps.is_available():
        #     self.pipeline.to('mps')

    async def translate(self, ocr_results: list[OcrResult]):
        #return [print(y) for y in self.pipeline([x.text for x in ocr_results])]


        return [TranslatorResult(y["translation_text"]) for y in self.pipeline([x.text for x in ocr_results])]

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
                default="Helsinki-NLP/opus-mt-ja-en",
            )
        ]
