import numpy as np
from manga_translator.core.plugin import (
    LanguagePluginSelectArgument,
    OCR,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
)


class DebugOCR(OCR):
    """Outputs the specified text"""

    def __init__(self, text="",language = 'en') -> None:
        super().__init__()
        self.to_write = text
        self.language = language

    async def extract(self, batch: list[np.ndarray]):
        return [OcrResult(text=self.to_write,language=self.to_write) for x in batch]

    @staticmethod
    def get_name() -> str:
        return "Debug"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="text", name="Debug Text", description="What to output"
            ),
            LanguagePluginSelectArgument(
                id="language",
                name="Output Language",
                description="The language the output text is",
                default="en"
            )
        ]
