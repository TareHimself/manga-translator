import numpy
from translator.core.plugin import (
    Ocr,
    OcrResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)


class CleanOcr(Ocr):
    """Cleans The Image i.e. does nothing"""

    def __init__(self) -> None:
        super().__init__()

    async def do_ocr(self, texts: list[numpy.ndarray]):
        return [OcrResult("", "") for _ in texts]

    @staticmethod
    def get_name() -> str:
        return "Clean Ocr"
