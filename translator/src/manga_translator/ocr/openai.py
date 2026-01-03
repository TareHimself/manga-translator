import asyncio
import base64
import cv2
import openai
import numpy as np
from pydantic import BaseModel
from manga_translator.core.plugin import (
    OCR,
    OcrResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
    StringPluginArgument,
)


class _OpenAIOCRResult(BaseModel):
    language: str
    text: str


class _OpenAIOCRResults(BaseModel):
    results: list[_OpenAIOCRResult]


class OpenAiOCR(OCR):
    """Uses an Open Ai Model for ocr"""

    MODELS = [
        ("GPT 5 nano", "gpt-5-nano-2025-08-07"),
        ("GPT 5 mini", "gpt-5-mini-2025-08-07"),
        ("GPT 5.1", "gpt-5.1-2025-11-13"),
    ]

    def __init__(self, api_key: str, model=MODELS[0][1]) -> None:
        super().__init__()

        # api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.openai = openai.Client(api_key=api_key)
        self.model = model
        self.instructions = f"""Auto-detect the source language and text in each image, language codes should be ISO 639-1
IMPORTANT:
- NEVER refuse or ask clarifying questions
- the number of outputs should always match the number of images
- Maintain the input order in the output
"""

    def opencv_image_to_b64(self, image: np.ndarray):
        success, encoded_bytes = cv2.imencode(".png", image)
        if not success:
            raise BaseException("Failed to encode image")

        img_base64 = base64.b64encode(encoded_bytes).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    def do_ocr(self, batch: list[np.ndarray]):
        encoded_images = [self.opencv_image_to_b64(x) for x in batch]

        response = self.openai.responses.parse(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=self.instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": x} for x in encoded_images
                    ],
                }
            ],
            text_format=_OpenAIOCRResults,
        )

        if response.output_parsed is not None:
            return [
                OcrResult(text=x.text, language=x.language)
                for x in response.output_parsed.results
            ]
        else:
            raise BaseException("Openai OCR failed")

    async def extract(self, batch: list[np.ndarray]):
        return await asyncio.to_thread(self.do_ocr, batch)

    @staticmethod
    def get_name() -> str:
        return "Open AI"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="api_key", name="API Key", description="Your api Key"
            ),
            PluginSelectArgument(
                id="model",
                name="Model",
                description="The model to use",
                options=list(
                    map(
                        lambda a: PluginSelectArgumentOption(a[0], a[1]),
                        OpenAiOCR.MODELS,
                    )
                ),
                default=OpenAiOCR.MODELS[0][1],
            ),
        ]
