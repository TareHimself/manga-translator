from typing import Any, Type
from manga_translator.core.plugin import BasePlugin, construct_plugin
from manga_translator.cleaning.get import get_cleaners
from manga_translator.detection.get import get_detectors
from manga_translator.segmentation.get import get_segmenters
from manga_translator.translation.get import get_translators
from manga_translator.ocr.get import get_ocrs
from manga_translator.drawing.get import get_drawers

_classes: list[Type[BasePlugin]] = [*get_cleaners(),*get_detectors(),*get_segmenters(),*get_translators(),*get_ocrs(),*get_drawers()]
_classes_dict = {}
for x in _classes:
    _classes_dict[x.__name__] = x

def get_all() -> dict[str,Type[BasePlugin]]:
    return _classes_dict

def construct_plugin_by_name(class_name: str,arguments: dict[str,Any]):
    return construct_plugin(_classes_dict[class_name],arguments)