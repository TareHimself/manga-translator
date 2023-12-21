import numpy
from translator.core.plugin import (
    Ocr,
    OcrResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)


class NoOcr(Ocr):
    """Does not perform any ocr on the image"""

    def __init__(self) -> None:
        super().__init__()

    async def do_ocr(self, batch: list[numpy.ndarray]):
        return [OcrResult("", "") for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "No Ocr"
