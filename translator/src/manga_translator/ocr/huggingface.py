import numpy as np
from manga_translator.core.plugin import PytorchDevicePluginSelectArgument, Translator
import torch
from transformers import pipeline
from manga_translator.core.plugin import (
    OCR,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
)
from manga_translator.utils import cv2_to_pil, get_default_torch_device
import asyncio

class HuggingFaceOCR(OCR):
    """Ocr using hugging face models"""

    def __init__(self, model_url: str,output_language: str = 'ja',device: torch.device = get_default_torch_device(),trust_remote_code = False) -> None:
        super().__init__()
        self.pipeline = pipeline("image-to-text", model=model_url, device=device,trust_remote_code = trust_remote_code)
        self.output_language = output_language

    def extract_with_model(self,  batch: list[np.ndarray]):
        result  = self.pipeline([cv2_to_pil(x) for x in batch])
        # should explore using more than just the first element in the future
        return [OcrResult(text=x[0]["generated_text"],language=self.output_language) for x in result]
    

    async def extract(self, batch: list[np.ndarray]):
        return await asyncio.to_thread(self.extract_with_model,batch)

    @staticmethod
    def get_name() -> str:
        return "Hugging Face"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="model_url",
                name="Model",
                description="The Hugging Face ocr model to use",
                default="TareHimself/manga-ocr-base",
            ),
            PytorchDevicePluginSelectArgument("device","Device")
        ]
