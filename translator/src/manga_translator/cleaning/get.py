from manga_translator.core.plugin import Cleaner
from manga_translator.cleaning.all_white_cleaner import AllWhiteCleaner
from manga_translator.cleaning.opencv import OpenCvCleaner
from manga_translator.cleaning.deepfillv2 import DeepFillV2Cleaner
from manga_translator.cleaning.lama import LamaCleaner
_data = list(
    filter(
        lambda a: a.is_valid(),
        [AllWhiteCleaner, OpenCvCleaner,DeepFillV2Cleaner,LamaCleaner],
    )
)


def get_cleaners() -> list[Cleaner]:
    return _data
