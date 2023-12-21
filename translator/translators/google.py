import os
from translator.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    PluginArgument,
    PluginTextArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)


class GoogleTranslateTranslator(Translator):
    """Translates using Google Translate"""

    def __init__(self, key_path="") -> None:
        super().__init__()

        self.key_path = key_path

        if len(self.key_path.strip()) > 0:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

            from google.cloud import translate_v2 as translate

            self.trans = translate.Client()
        else:
            self.trans = None

    
    async def translate(self, batch: list[OcrResult]):
        if self.trans is None:
            return [TranslatorResult("Invalid Key Path") for _ in batch]

        return [TranslatorResult(
            self.trans.translate(
                x.text,
                source_language=x.language,
                target_language="en",
            )["translatedText"]
        ) for x in batch]

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            PluginTextArgument(
                id="key_path",
                name="Service Account Key Path",
                description="Path to google application credentials json",
            )
        ]

    @staticmethod
    def get_name() -> str:
        return "Google Cloud"
