import asyncio
import httpx
import base64
from io import BytesIO
from more_itertools import chunked
import numpy as np
from comic_localizer.core.plugin import (
    OCR,
    OcrResult,
    PluginArgument,
    StringPluginArgument,
)
from comic_localizer.utils import (
    cv2_to_pil,
    get_default_language,
    lingua_lang_to_lang_code,
)
from lingua import Language, LanguageDetectorBuilder
from typing import Optional
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain


class MinstralOCR(OCR):
    """Uses Minstral to perform OCR https://mistral.ai/news/mistral-ocr , requires an API key"""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-ocr-latest",
        detectable_languages: Optional[list[Language]] = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.language_detector = (
            LanguageDetectorBuilder.from_languages(*detectable_languages).build()
            if detectable_languages is not None
            else LanguageDetectorBuilder.from_all_spoken_languages().build()
        )
        self.parser = MarkdownIt(renderer_cls=RendererPlain)
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            base_url="https://api.mistral.ai/v1",
        )

    @staticmethod
    def make_pdf_b64(batch: list[np.ndarray]):
        images_as_pil = [cv2_to_pil(x) for x in batch]
        data = BytesIO()
        images_as_pil[0].save(
            data, format="PDF", save_all=True, append_images=images_as_pil[1:]
        )

        return base64.b64encode(data.getbuffer()).decode("utf-8")

    async def extract(self, batch: list[np.ndarray]):
        ocr_results: list[OcrResult] = []

        for images in chunked(batch, 200):
            # sending multiple images as a PDF works best, might try this for GPT
            b64_data = await asyncio.to_thread(MinstralOCR.make_pdf_b64, images)

            resp = await self.client.post(
                "/ocr",
                json={
                    "model": self.model,
                    "document": {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{b64_data}",
                    },
                },
            )

            resp.raise_for_status()

            result = resp.json()

            offset = len(ocr_results)

            # not sure if pages are 1 for 1
            texts = [x["markdown"] for x in result["pages"]]
            languages = await asyncio.to_thread(
                self.language_detector.detect_languages_in_parallel_of, texts
            )
            text_dict = {}

            for page, lang in zip(result["pages"], languages):
                text_dict[page["index"]] = (
                    self.parser.render(page["markdown"]),
                    get_default_language()
                    if lang is None
                    else lingua_lang_to_lang_code(lang),
                )

            for _ in range(len(images)):
                ocr_results.append(OcrResult())

            for key in text_dict:
                text, lang_code = text_dict[key]
                ocr_results[offset + key].text = text
                ocr_results[offset + key].language = lang_code

        return ocr_results

    @staticmethod
    def get_name() -> str:
        return "Minstral OCR"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="api_key", name="API Key", description="Your api key"
            ),
            StringPluginArgument(
                id="model",
                name="Model",
                description="The Minstral model to use",
                default="mistral-ocr-latest",
            ),
        ]
