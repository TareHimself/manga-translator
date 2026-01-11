from manga_translator.core.plugin import ColorDetector
from manga_translator.color_detection.openai import OpenAiColorDetector

_color_detection_data = list(
    filter(
        lambda a: a.is_valid(),
        [ColorDetector, OpenAiColorDetector],
    )
)


def get_color_detectors() -> list[ColorDetector]:
    return _color_detection_data
