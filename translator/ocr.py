import numpy
from translator.utils import cv2_to_pil


class OcrResult:
    def __init__(self, text: str = "Sample", language: str = "en") -> None:
        self.text = text
        self.language = language


class BaseOcr:
    """Base Class for all OCR classes"""

    def __init__(self) -> None:
        pass

    def __call__(self, text: numpy.ndarray) -> OcrResult:
        return self.do_ocr(text)

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult()


class MangaOcr(BaseOcr):
    """Only Supports Japanese"""

    def __init__(self) -> None:
        from manga_ocr import MangaOcr as MangaOcrPackage

        super().__init__()
        self.manga_ocr = MangaOcrPackage()

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult(self.manga_ocr(cv2_to_pil(text)), "ja")


def get_ocr():
    return [BaseOcr, MangaOcr]
