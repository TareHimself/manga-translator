from translator.core.plugin import Translator
from translator.translators.deepl import DeepLTranslator
from translator.translators.google import GoogleTranslateTranslator
from translator.translators.hugging_face import HuggingFace
from translator.translators.debug import DebugTranslator
from translator.translators.openai import OpenAiTranslator

def get_translators() -> list[Translator]:
    return list(filter(lambda a: a.is_valid(), [
        HuggingFace,
        DeepLTranslator,
        GoogleTranslateTranslator,
        DebugTranslator,
        OpenAiTranslator
    ]))