import numpy
import traceback
import os
from translator.utils import cv2_to_pil, lang_code_to_name, simplify_lang_code
from translator.core.plugin import BasePlugin, PluginArgument, PluginSelectArgument, PluginSelectArgumentOption


class OcrResult:
    def __init__(self, text: str = "", language: str = "en") -> None:
        self.text = text
        self.language = language


class BaseOcr(BasePlugin):
    """Always outputs \"Sample\""""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text: numpy.ndarray) -> OcrResult:
        return self.do_ocr(text)

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult("Sample")

    @staticmethod
    def get_name() -> str:
        return "Base Ocr"


class CleanOcr(BaseOcr):
    """Cleans The Image i.e. does nothing"""

    def __init__(self) -> None:
        super().__init__()

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult("", "")

    @staticmethod
    def get_name() -> str:
        return "Clean Ocr"


class MangaOcr(BaseOcr):
    """Only Supports Japanese"""

    def __init__(self) -> None:
        from manga_ocr import MangaOcr as MangaOcrPackage

        super().__init__()
        self.manga_ocr = MangaOcrPackage()

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult(self.manga_ocr(cv2_to_pil(text)), "ja")

    @staticmethod
    def get_name() -> str:
        return "Manga Ocr"


class EasyOcr(BaseOcr):
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
        "vi"
    ]

    def __init__(self, lang=languages[0]) -> None:
        import easyocr

        super().__init__()
        self.easy = easyocr.Reader([lang])
        self.language = lang

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult(text=self.easy.readtext(text, detail=0, paragraph=True)[0],
                         language=self.language)  # self.language)

    @staticmethod
    def get_name() -> str:
        return "Easy Ocr"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        options = list(filter(lambda a: a.name is not None,
                              [PluginSelectArgumentOption(name=lang_code_to_name(lang), value=lang) for lang in
                               EasyOcr.languages]))

        return [PluginSelectArgument(id="lang",
                                     name="Language",
                                     description="The language to detect",
                                     options=options, default=options[0].value)]


class TesseractOcr(BaseOcr):
    """Supports all the languages listed"""

    default_language = "jpn"
    tessaract_path = os.path.abspath("C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

    def __init__(self, language=default_language) -> None:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TesseractOcr.tessaract_path
        super().__init__()
        self.tesseract = pytesseract
        self.language = language

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult(text=self.tesseract.image_to_string(text, lang=self.language),
                         language=simplify_lang_code(self.language))

    @staticmethod
    def get_name() -> str:
        return "Tesseract Ocr"

    @staticmethod
    def is_valid() -> bool:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = TesseractOcr.tessaract_path
            pytesseract.get_tesseract_version()
            return True
        except:
            traceback.print_exc()
            return False

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TesseractOcr.tessaract_path
        languages = pytesseract.get_languages()
        languages.sort()
        if "jpn" in languages:
            languages.remove("jpn")
            languages.insert(0, "jpn")

        options = list(filter(lambda a: a.name is not None,
                              [PluginSelectArgumentOption(name=lang_code_to_name(lang), value=lang) for lang in
                               languages]))

        return [PluginSelectArgument(id="language",
                                     name="Language",
                                     description="The language to detect",
                                     options=options, default=languages[0])]
        # options = [PluginSelectArgumentOption(name=value,value=key) for key, value in EasyOcr.languages.items()]
        # options.sort(key= lambda a: "." if a.value == "ja" else a.name)
        # return [PluginSelectArgument(name="language",
        #                              description="The language to detect",
        #                              options=options)]


def get_ocr() -> list[BaseOcr]:
    return list(filter(lambda a: a.is_valid(), [BaseOcr, CleanOcr, MangaOcr, EasyOcr, TesseractOcr]))
