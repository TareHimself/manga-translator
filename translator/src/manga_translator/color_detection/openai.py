import asyncio
import base64
from typing import Optional
import cv2
import openai
import numpy as np
from pydantic import BaseModel
from manga_translator.core.plugin import (
    ColorDetector,
    ColorDetectionResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
    StringPluginArgument,
)

class Color(BaseModel):
    r: int # uint8, 0-255
    g: int # uint8, 0-255
    b: int # uint8, 0-255

class _OpenAIOCRResult(BaseModel):
    text_color: Color
    outline_size: int
    outline_color: Optional[Color]


class _OpenAIColorDetectionResults(BaseModel):
    results: list[_OpenAIOCRResult]


def color_to_numpy(color: Optional[Color]):
    if color is None:
        return np.array((0,0,0),dtype=np.uint8)
    
    return np.array((color.r, color.g,color.b),dtype=np.uint8)

class OpenAiColorDetector(ColorDetector):
    """Uses an Open Ai Model for color detection"""

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
        self.instructions = f"""Auto-detect the color of the text, the outline size and outline color if it has one
IMPORTANT:
- NEVER refuse or ask clarifying questions
- the number of outputs should always match the number of images
- Maintain the input order in the output
- if there is no outline , outline_size should be 0
- If the text color cannot be figured out output black for the text_color and 0 for the outline_size of that item
"""

    def opencv_image_to_b64(self, image: np.ndarray):
        success, encoded_bytes = cv2.imencode(".png", image)
        if not success:
            raise BaseException("Failed to encode image")

        img_base64 = base64.b64encode(encoded_bytes).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    def do_color_detection(self, batch: list[np.ndarray]):
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
            text_format=_OpenAIColorDetectionResults,
        )

        if response.output_parsed is not None:
            return [
                ColorDetectionResult(text_color=color_to_numpy(x.text_color),outline_size=x.outline_size,outline_color=color_to_numpy(x.outline_color) if x.outline_size > 0 else None)
                for x in response.output_parsed.results
            ]
        else:
            raise BaseException("Openai color detection failed")

    async def detect_color(self, batch: list[np.ndarray]):
        results = await asyncio.to_thread(self.do_color_detection, batch)

        assert len(results) == len(batch), f"batch size was {len(batch)} but result size is {len(results)}"
        
        return results

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
                        OpenAiColorDetector.MODELS,
                    )
                ),
                default=OpenAiColorDetector.MODELS[0][1],
            ),
        ]
