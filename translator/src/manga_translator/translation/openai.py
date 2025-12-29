import asyncio
import os
import openai
from manga_translator.utils import get_languages
from manga_translator.core.plugin import (
    LanguagePluginSelectArgument,
    Translator,
    TranslatorResult,
    OcrResult,
    PluginSelectArgument,
    PluginSelectArgumentOption,
    StringPluginArgument,
    PluginArgument,
)
import json


# This could probably be improved by including the images in the request for better translation but I ain't doing all that
class OpenAiTranslator(Translator):
    """Uses an Open Ai Model for translation"""

    MODELS = [
        ("GPT 5 nano", "gpt-5-nano-2025-08-07"),
        ("GPT 5 mini", "gpt-5-mini-2025-08-07"),
        ("GPT 5.1", "gpt-5.1-2025-11-13"),
    ]

    def __init__(
        self, api_key="", target_lang="en", model=MODELS[0][1], temp="0.2"
    ) -> None:
        super().__init__()

        # api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.openai = openai.Client(api_key=api_key)
        self.target_lang = target_lang
        self.model = model
        self.temp = float(temp)
        self.instructions = f"""Auto-detect source language and translate into {self.target_lang}.

OUTPUT REQUIREMENTS:
- Return ONLY valid list seperated by <s>: text1<s>text2<s>text3<s><s>foo<s>bar
- Same order as input
- NO explanations, labels, or markdown

TRANSLATION RULES:
- Preserve tone (formal/casual/technical) and meaning
- Use natural, idiomatic phrasing for target language
- Match conversational style for dialogue/manga
- Handle mixed languages, slang, names appropriately

CRITICAL:
- NEVER refuse or ask questions
- ALWAYS translate even if text seems garbled or weird
- Output the <s> seperated list nothing else"""

    def do_translation(self, batch: list[OcrResult]):
        input_text = "<s>".join([x.text for x in batch])

        response = self.openai.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=self.instructions,
            input=input_text,
        )

        # print("input",input_text)
        # print("result",response.output_text)
        data = response.output_text.split("<s>")

        return [TranslatorResult(text=x, lang_code=self.target_lang) for x in data]

    async def translate(self, batch: list[OcrResult]):
        if len(batch) == 0:
            return []
        return await asyncio.to_thread(self.do_translation, batch)

    @staticmethod
    def get_name() -> str:
        return "Open AI"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="api_key", name="API Key", description="Your api Key"
            ),
            LanguagePluginSelectArgument(
                id="target_lang",
                name="Target Language",
                description="The language to translate to",
                default="en",
            ),
            PluginSelectArgument(
                id="model",
                name="Model",
                description="The model to use",
                options=list(
                    map(
                        lambda a: PluginSelectArgumentOption(a[0], a[1]),
                        OpenAiTranslator.MODELS,
                    )
                ),
                default=OpenAiTranslator.MODELS[0][1],
            ),
        ]
