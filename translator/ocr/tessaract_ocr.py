import numpy
import os
from translator.utils import lang_code_to_name, simplify_lang_code, resize_and_pad
from translator.core.plugin import (
    Ocr,
    OcrResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)
from translator.utils import display_image, resize_and_pad, ensure_gray, adjust_contrast_brightness

class TesseractOcr(Ocr):
    """Supports all the languages listed"""

    default_language = "jpn"
    tessaract_path = os.path.abspath("C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

    def __init__(self, language=default_language) -> None:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = TesseractOcr.tessaract_path
        super().__init__()
        self.tesseract = pytesseract
        self.language = language
    

    async def do_ocr(self, batch: list[numpy.ndarray]):

        return [OcrResult(
            text=self.tesseract.image_to_string(x, lang=self.language),
            language=simplify_lang_code(self.language),
        ) for x in batch]

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
            print(
                "Either pytesseract is having an error or you do not have it installed!."
            )
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

        options = list(
            filter(
                lambda a: a.name is not None,
                [
                    PluginSelectArgumentOption(name=lang_code_to_name(lang), value=lang)
                    for lang in languages
                ],
            )
        )

        return [
            PluginSelectArgument(
                id="language",
                name="Language",
                description="The language to detect",
                options=options,
                default=languages[0],
            )
        ]
        # options = [PluginSelectArgumentOption(name=value,value=key) for key, value in EasyOcr.languages.items()]
        # options.sort(key= lambda a: "." if a.value == "ja" else a.name)
        # return [PluginSelectArgument(name="language",
        #                              description="The language to detect",
        #                              options=options)]
