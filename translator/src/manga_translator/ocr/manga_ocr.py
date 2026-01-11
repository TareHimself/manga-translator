import re
import jaconv
import numpy as np
from manga_translator.core.plugin import PytorchDevicePluginSelectArgument, Translator
import torch
from transformers import pipeline
from manga_translator.core.plugin import (
    OCR,
    OcrResult,
    PluginArgument,
)
from manga_translator.utils import cv2_to_pil, get_default_torch_device
import asyncio
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin
from manga_translator.utils import perf_async

class _MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    pass

class MangaOCR(OCR):
    """Ocr using kha-white/manga-ocr-base"""

    def __init__(self, model_url: str = "kha-white/manga-ocr-base",device: torch.device = get_default_torch_device()) -> None:
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_url,use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = _MangaOcrModel.from_pretrained(model_url)
        self.model.to(device)
        self.device = device
        self.output_language = "ja"

    @staticmethod
    def post_process(text):
        text = "".join(text.split())
        text = text.replace("…", "...")
        text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
        text = jaconv.h2z(text, ascii=True, digit=True)

        return text
    
    
    def _preprocess(self, images):
        images = [cv2_to_pil(x).convert("L").convert("RGB") for x in images]
        result = self.processor(images, return_tensors="pt")
        return result["pixel_values"].to(self.device)

    def extract_with_model(self,  batch: list[np.ndarray]):
        with torch.inference_mode():
            x = self._preprocess(batch)
            x = self.model.generate(x, max_length=300).cpu()
            y = [self.tokenizer.decode(z, skip_special_tokens=True) for z in x]

            # should explore using more than just the first element in the future
            return [OcrResult(text=x,language=self.output_language) for x in map(self.post_process,y)]
    
    @perf_async
    async def extract(self, batch: list[np.ndarray]):
        return await asyncio.to_thread(self.extract_with_model,batch)

    @staticmethod
    def get_name() -> str:
        return "Manga (by kha white)"
    
    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            PytorchDevicePluginSelectArgument("device","Device")
        ]
