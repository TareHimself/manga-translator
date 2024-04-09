import numpy
from translator.utils import cv2_to_pil, lang_code_to_name
from translator.core.plugin import (
    Ocr,
    OcrResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)


class EasyOcr(Ocr):
    """Supports all the languages listed"""

    languages = [
        "ja",
        "abq",
        "ady",
        "af",
        "ang",
        "ar",
        "as",
        "ava",
        "az",
        "be",
        "bg",
        "bh",
        "bho",
        "bn",
        "bs",
        "ch_sim",
        "ch_tra",
        "che",
        "cs",
        "cy",
        "da",
        "dar",
        "de",
        "en",
        "es",
        "et",
        "fa",
        "fr",
        "ga",
        "gom",
        "hi",
        "hr",
        "hu",
        "id",
        "inh",
        "is",
        "it",
        "kbd",
        "kn",
        "ko",
        "ku",
        "la",
        "lbe",
        "lez",
        "lt",
        "lv",
        "mah",
        "mai",
        "mi",
        "mn",
        "mr",
        "ms",
        "mt",
        "ne",
        "new",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "ru",
        "rs_cyrillic",
        "rs_latin",
        "sck",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "ta",
        "tab",
        "te",
        "th",
        "tjk",
        "tl",
        "tr",
        "ug",
        "uk",
        "ur",
        "uz",
        "vi",
    ]

    def __init__(self, lang=languages[0]) -> None:
        import easyocr

        super().__init__()
        self.easy = easyocr.Reader([lang])
        self.language = lang

    async def do_ocr(self, batch: list[numpy.ndarray]):
        result = []
        for x in batch:
            read_text_result = self.easy.readtext(x, detail=0, paragraph=True)
            text = read_text_result[0] if read_text_result else ''
            result.append(OcrResult(text=text, language=self.language))
        return result

    @staticmethod
    def get_name() -> str:
        return "Easy Ocr"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        options = list(
            filter(
                lambda a: a.name is not None,
                [
                    PluginSelectArgumentOption(name=lang_code_to_name(lang), value=lang)
                    for lang in EasyOcr.languages
                ],
            )
        )

        return [
            PluginSelectArgument(
                id="lang",
                name="Language",
                description="The language to detect",
                options=options,
                default=options[0].value,
            )
        ]
