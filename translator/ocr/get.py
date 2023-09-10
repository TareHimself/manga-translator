from translator.core.plugin import Ocr
from translator.ocr.clean import CleanOcr
from translator.ocr.manga import MangaOcr
from translator.ocr.easy_ocr import EasyOcr
from translator.ocr.tessaract_ocr import TesseractOcr

def get_ocr() -> list[Ocr]:
    return list(filter(lambda a: a.is_valid(), [CleanOcr, MangaOcr, EasyOcr, TesseractOcr]))