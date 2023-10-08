from translator.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    PluginTextArgument,
    PluginArgument,
)


class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self, text="") -> None:
        super().__init__()
        self.to_write = text

    async def translate(self, ocr_results: list[OcrResult]):
        return [TranslatorResult(self.to_write) for _ in ocr_results]

    @staticmethod
    def get_name() -> str:
        return "Debug Translator"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            PluginTextArgument(
                id="text", name="Debug Text", description="What to write"
            )
        ]
