from manga_translator.core.plugin import LanguagePluginSelectArgument, PytorchDevicePluginSelectArgument, Translator
import torch
from transformers import pipeline
from manga_translator.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
)
import asyncio
from manga_translator.utils import get_default_torch_device

class HuggingFaceTranslator(Translator):
    """Translates using hugging face models"""

    def __init__(self, model_url: str = "Helsinki-NLP/opus-mt-ja-en", input_language: str = 'ja',output_language: str = 'en',device: torch.device = get_default_torch_device()) -> None:
        super().__init__()
        self.pipeline = pipeline("translation", model=model_url, device=device,src_lang=input_language,tgt_lang=output_language)
        self.input_language = input_language
        self.output_language = output_language
        # if torch.cuda.is_available():
        #     self.pipeline.cuda()
        # elif torch.backends.mps.is_available():
        #     self.pipeline.to('mps')

    def predict(self,batch: list[OcrResult]):
        with torch.inference_mode():
            results = self.pipeline([x.text for x in batch])
            return results
    async def translate(self, batch: list[OcrResult]):
        #return [print(y) for y in self.pipeline([x.text for x in batch])]
        results = await asyncio.to_thread(self.predict,batch)

        return [TranslatorResult(y["translation_text"],lang_code=self.output_language) for y in results]

    @staticmethod
    def get_name() -> str:
        return "Hugging Face"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        
        return [
            StringPluginArgument(
                id="model_url",
                name="Model",
                description="The Hugging Face translation model to use",
                default="Helsinki-NLP/opus-mt-ja-en",
            ),
             LanguagePluginSelectArgument(
                id="input_language",
                name="Input Language",
                description="The language to translate from",
                default="ja",
            ),
             LanguagePluginSelectArgument(
                id="output_language",
                name="Output Language",
                description="The language to translate to",
                default="en",
            ),
            PytorchDevicePluginSelectArgument("device","Device")
        ]
