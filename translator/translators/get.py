from translator.core.plugin import Translator
from translator.translators.deepl import DeepLTranslator
from translator.translators.google import GoogleTranslateTranslator
from translator.translators.hugging_face import HuggingFace
from translator.translators.debug import DebugTranslator
from translator.translators.openai import OpenAiTranslator
from translator.translators.gemini import GeminiTranslator


def get_translators() -> list[Translator]:
    return list(
        filter(
            lambda a: a.is_valid(),
            [
                DebugTranslator,
                HuggingFace,
                DeepLTranslator,
                GoogleTranslateTranslator,
                OpenAiTranslator,
                GeminiTranslator
            ],
        )
    )
