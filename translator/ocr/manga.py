import numpy
from translator.utils import cv2_to_pil
from translator.core.plugin import Ocr,OcrResult




class MangaOcr(Ocr):
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