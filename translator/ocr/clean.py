import numpy
from translator.core.plugin import Ocr,OcrResult, PluginArgument, PluginSelectArgument, PluginSelectArgumentOption





class CleanOcr(Ocr):
    """Cleans The Image i.e. does nothing"""

    def __init__(self) -> None:
        super().__init__()

    def do_ocr(self, text: numpy.ndarray):
        return OcrResult("", "")

    @staticmethod
    def get_name() -> str:
        return "Clean Ocr"

