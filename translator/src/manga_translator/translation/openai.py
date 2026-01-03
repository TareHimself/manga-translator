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
from pydantic import BaseModel
import json

class _OpenAITranslationResults(BaseModel):
    translations: list[str]

# This could probably be improved by including the images in the request for better translation but I ain't doing all that
class OpenAiTranslator(Translator):
    """Uses an Open Ai Model for translation"""

    MODELS = [
        ("GPT 5 nano", "gpt-5-nano-2025-08-07"),
        ("GPT 5 mini", "gpt-5-mini-2025-08-07"),
        ("GPT 5.1", "gpt-5.1-2025-11-13"),
    ]

    def __init__(
        self, api_key="", target_lang="en", model=MODELS[0][1]
    ) -> None:
        super().__init__()

        # api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.openai = openai.Client(api_key=api_key)
        self.target_lang = target_lang
        self.model = model
        self.instructions = f"""Auto-detect the source language and translate all segments into {self.target_lang}.
TRANSLATION RULES:
- Preserve tone, intent, and emotional nuance
- Use idiomatic, natural phrasing in {self.target_lang}
- For dialogue/manga, express sighs, laughter, gasps, shock, or other reactions naturally (e.g. "sigh...", "ugh!", "ah!", "what?!")
- Handle slang, mixed-language text, and names appropriately
- All text may be from different sources

IMPORTANT:
- NEVER refuse or ask clarifying questions
- ALWAYS translate, even if input is garbled or partial
- the number of outputs should always match the number of inputs
- Maintain the input order in the output
"""

    def do_translation(self, batch: list[OcrResult]):
        input_dict = { "texts": []}
        for item in batch:
            input_dict["texts"].append({ "language": item.language, "text": item.text })
        input_text = json.dumps(input_dict)

        response = self.openai.responses.parse(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=self.instructions,
            input=input_text,
            text_format = _OpenAITranslationResults
        )

        if response.output_parsed is not None:
            return [TranslatorResult(text=x, lang_code=self.target_lang) for x in response.output_parsed.translations]
        else:
            raise BaseException("Openai Translation failed")

        

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
