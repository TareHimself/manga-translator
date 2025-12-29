from manga_translator.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
    LanguagePluginSelectArgument,
)


class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self, text="", language="en") -> None:
        super().__init__()
        self.to_write = text
        self.language = language

    async def translate(self, batch: list[OcrResult]):
        return [TranslatorResult(self.to_write, lang_code=self.language) for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Custom Text"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="text", name="Debug Text", description="What to write"
            ),
            LanguagePluginSelectArgument(
                id="language",
                name="Output Language",
                description="The language the output text is",
                default="en",
            ),
        ]
