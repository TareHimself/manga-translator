from manga_translator.core.plugin import Translator
from manga_translator.translation.deepl import DeepLTranslator
from manga_translator.translation.huggingface import HuggingFaceTranslator
from manga_translator.translation.openai import OpenAiTranslator
from manga_translator.translation.debug import DebugTranslator
from manga_translator.translation.pipe import PipeTranslator

_data = list(
    filter(
        lambda a: a.is_valid(),
        [
            DebugTranslator,
            HuggingFaceTranslator,
            DeepLTranslator,
            OpenAiTranslator,
            PipeTranslator,
        ],
    )
)


def get_translators() -> list[Translator]:
    return _data
