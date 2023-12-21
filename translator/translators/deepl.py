import requests
import traceback
import asyncio
import json
from requests.utils import requote_uri
import aiohttp
from translator.core.plugin import (
    Translator,
    OcrResult,
    TranslatorResult,
    PluginArgument,
    PluginTextArgument,
)


class DeepLTranslator(Translator):
    """The Best after GPT but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_token=None) -> None:
        super().__init__()
        self.auth_token = auth_token

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            PluginTextArgument(
                id="auth_token", name="Auth Token", description="DeepL Api Auth Token"
            )
        ]
    
    async def do_api(self,result: OcrResult):
        if self.auth_token is None or len(self.auth_token.strip()) == 0:
            return TranslatorResult("Need DeepL Auth")

        if len(result.text.strip()) == 0:
            return TranslatorResult("")

        if result.language == "ja":
            try:
                data = [
                    ("target_lang", "EN-US"),
                    ("source_lang", "JA"),
                    ("text", result.text),
                ]
                uri = f"https://api-free.deepl.com/v2/translate?{'&'.join([f'{data[i][0]}={data[i][1]}' for i in range(len(data))])}"
                uri = requote_uri(uri)

                async with aiohttp.ClientSession() as session:
                    async with session.post(uri,headers={
                            "Authorization": f"DeepL-Auth-Key {self.auth_token}",
                            "Content-Type": "application/x-www-form-urlencoded",
                        }) as response:

                        data = await response.json()
                        
                        return TranslatorResult(
                            data["translations"][0]["text"],lang_code="ja"
                        )
            except:
                traceback.print_exc()
                return TranslatorResult("Failed To Get Translation")
        else:
            return TranslatorResult("Language not supported")

    async def translate(self, batch: list[OcrResult]):
        return await asyncio.gather(*[self.do_api(x) for x in batch])
        

    @staticmethod
    def get_name() -> str:
        return "DeepL"
