import requests
import traceback
import asyncio
import json
from requests.utils import requote_uri
from manga_translator.core.plugin import (
    Translator,
    OcrResult,
    TranslatorResult,
    PluginArgument,
    StringPluginArgument,
)
import deepl


class DeepLTranslator(Translator):
    """The Best after GPT but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_key=None) -> None:
        super().__init__()
        self.client = deepl.DeepLClient(auth_key)

    def do_api(self, batch: list[OcrResult]):
        try:
            results = self.client.translate_text(
                [x.text for x in batch],
                target_lang="EN-US",
            )

            return [TranslatorResult(text=x.text) for x in results]
        except:
            traceback.print_exc()
            return [TranslatorResult("Failed To Get Translation") for _ in batch]

    async def translate(self, batch: list[OcrResult]):
        return await asyncio.to_thread(self.do_api, batch)

    @staticmethod
    def get_name() -> str:
        return "DeepL"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="auth_key", name="Auth Token", description="DeepL Api Auth Key"
            )
        ]
