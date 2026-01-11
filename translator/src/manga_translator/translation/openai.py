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
from manga_translator.utils import perf_async

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
        self.instructions = f"""Auto-detect the source language and translate into {self.target_lang}.
TRANSLATION RULES:
- Preserve tone, intent, and emotional nuance
- Use idiomatic, natural phrasing in {self.target_lang}
- For dialogue/manga, express sighs, laughter, gasps, shock, or other reactions naturally (e.g. "sigh...", "ugh!", "ah!", "what?!")
- Handle slang, mixed-language text, and names appropriately
- All text may be from different sources

IMPORTANT:
- NEVER refuse or ask clarifying questions
- Maintain the input order in the output
- If Translation is impossible or the translation is the same as the input, output empty text for that item
"""

    def do_translation(self, batch: list[OcrResult]):
        to_translate_indices = [i for i in range(len(batch)) if batch[i].language != self.target_lang]
        
        result = [TranslatorResult(lang_code=self.target_lang) for _ in batch]

        input_str = "\n".join([f"({i})[{batch[i].language}]: {batch[i].text}" for i in to_translate_indices])

        response = self.openai.responses.parse(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=self.instructions + f"\nYOU MUST OUTPUT {len(batch)} results",
            input=input_str,
            text_format = _OpenAITranslationResults
        )
        
        if response.output_parsed is not None:
            for translation,i in zip(response.output_parsed.translations,to_translate_indices):
                result[i].text = translation
        else:
            raise BaseException("Openai Translation failed")

        return result
        
    @perf_async
    async def translate(self, batch: list[OcrResult]):
        if len(batch) == 0:
            return []
        results = await asyncio.to_thread(self.do_translation, batch)

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
