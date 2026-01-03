from manga_translator.core.plugin import OCR
from manga_translator.ocr.debug import DebugOCR
from manga_translator.ocr.huggingface import HuggingFaceOCR
from manga_translator.ocr.manga_ocr import MangaOCR
from manga_translator.ocr.openai import OpenAiOCR

_ocr_data = list(
    filter(
        lambda a: a.is_valid(),
        [
            DebugOCR,
            HuggingFaceOCR,
            MangaOCR,
            OpenAiOCR
        ],
    )
)

def get_ocrs() -> list[OCR]:
    return _ocr_data
