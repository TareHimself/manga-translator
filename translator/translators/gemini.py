import asyncio
import aiohttp
import traceback
import json
from translator.core.plugin import (
    PluginSelectArgument,
    PluginSelectArgumentOption,
    Translator,
    OcrResult,
    TranslatorResult,
    PluginArgument,
    PluginTextArgument,
)
from translator.utils import get_languages

class GeminiTranslator(Translator):
    def __init__(self, api_key="",target_lang="en") -> None:
        super().__init__()
        self.api_key = api_key
        self.target_lang = target_lang

    MAX_DEPTH = 5
    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        languages = get_languages()
        languages.sort(key=lambda a: a[0].lower())
        options = list(map(lambda a: PluginSelectArgumentOption(a[0], a[1]), languages))
        
        return [
            PluginTextArgument(
                id="api_key", name="Api Key", description="Gemini Api Key"
            ),
            PluginSelectArgument(
                id="target_lang",
                name="Target Language",
                description="The language to translate to",
                options=options,
                default="en",
            ),
        ]
    
    async def do_api(self,result: OcrResult,depth = 0):
        if self.api_key is None or len(self.api_key.strip()) == 0:
            return TranslatorResult("Need Gemini api key")

        if len(result.text.strip()) == 0:
            return TranslatorResult("")
        

        message = f"{result.language.upper()} to {self.target_lang.upper()}\n{result.text}"
        body = {
            "contents" : [
                {
                    "role" : "user",
                    "parts" : [
                        {
                            "text" : "EN to JA\nHello World"
                        }
                    ]
                },
                {
                    "role" : "model",
                    "parts" : [
                        {
                            "text" : "こんにちは世界"
                        }
                    ]
                },
                {
                    "role" : "user",
                    "parts" : [
                        {
                            "text" : message
                        }
                    ]
                }
            ]
        }

        uri = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"

        try:

            async with aiohttp.ClientSession() as session:
                async with session.post(uri,headers={
                        "Content-Type": "application/json",
                    },data=json.dumps(body)) as response:

                    data = await response.json()

                    if "candidates" not in data.keys():
                        if "error" in data.keys():
                            if depth < GeminiTranslator.MAX_DEPTH:
                                asyncio.sleep(0.1)
                                return self.do_api(result,depth + 1)
                            else:
                                return TranslatorResult("Gemini failed to translate for safety reasons")
                        elif "promptFeedback" in data.keys():
                            print("Gemini failed to translate for safety reasons :",data["promptFeedback"])
                            return TranslatorResult("Gemini failed to translate for safety reasons")
                        else:
                            print(data)
                            return TranslatorResult("Unknown Gemini Error, Check console")


                        

                    return TranslatorResult(
                        data["candidates"][0]["content"]["parts"][0]["text"],lang_code=self.target_lang
                    )
        except:
            traceback.print_exc()
            return TranslatorResult("Failed To Get Translation")

    async def translate(self, batch: list[OcrResult]):
        return await asyncio.gather(*[self.do_api(x) for x in batch])
        

    @staticmethod
    def get_name() -> str:
        return "Gemini"