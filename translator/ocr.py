import numpy
from translator.utils import cv2_to_pil
from translator.plugin import BasePlugin

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
    
    def get_name() -> str:
        return "Base Ocr"
    
class CleanOcr(BaseOcr):
    """Cleans The Image i.e. does nothing"""

    def __init__(self) -> None:
        super().__init__()

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult("", "")
    
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
    
    def get_name() -> str:
        return "Manga Ocr"


def get_ocr()  -> list[BaseOcr]:
    return [BaseOcr,CleanOcr, MangaOcr]
