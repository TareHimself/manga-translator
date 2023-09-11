from translator.core.plugin import Cleaner
from translator.cleaners.deepfillv2 import DeepFillV2Cleaner
from translator.cleaners.lama import LamaCleaner


def get_cleaners() -> list[Cleaner]:
    return [DeepFillV2Cleaner, LamaCleaner]
