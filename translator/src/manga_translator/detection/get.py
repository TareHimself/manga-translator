from manga_translator.core.plugin import Detector
from manga_translator.detection.yolo import YoloDetector

_data = list(
    filter(
        lambda a: a.is_valid(),
        [YoloDetector],
    )
)


def get_detectors() -> list[Detector]:
    return _data
    
