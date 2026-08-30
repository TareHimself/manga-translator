from typing import Sequence
from comic_localizer.core.plugin import Translator
from comic_localizer.translation.ctranslate import CTranslateTranslator
from comic_localizer.translation.deepl import DeepLTranslator
from comic_localizer.translation.huggingface import HuggingFaceTranslator
from comic_localizer.translation.openai import OpenAiTranslator
from comic_localizer.translation.debug import DebugTranslator
from comic_localizer.translation.pipe import PipeTranslator

_translator_data = list(
    filter(
        lambda a: a.is_valid(),
        [
            DebugTranslator,
            CTranslateTranslator,
            DeepLTranslator,
            HuggingFaceTranslator,
            OpenAiTranslator,
            PipeTranslator,
        ],
    )
)


def get_translators() -> Sequence[type[Translator]]:
    return _translator_data
