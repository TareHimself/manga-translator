from typing import Any, Type
from manga_translator.core.plugin import BasePlugin, construct_plugin
from manga_translator.cleaning.get import get_cleaners
from manga_translator.detection.get import get_detectors
from manga_translator.segmentation.get import get_segmenters
from manga_translator.translation.get import get_translators
from manga_translator.ocr.get import get_ocrs
from manga_translator.drawing.get import get_drawers
from manga_translator.pipelines.image_to_image import ImageToImagePipeline
import yaml

_classes: list[Type[BasePlugin]] = [*get_cleaners(),*get_detectors(),*get_segmenters(),*get_translators(),*get_ocrs(),*get_drawers()]
_classes_dict = {}
for x in _classes:
    _classes_dict[x.__name__] = x

def get_all() -> dict[str,Type[BasePlugin]]:
    return _classes_dict

def construct_plugin_by_name(class_name: str,arguments: dict[str,Any]):
    return construct_plugin(_classes_dict[class_name],arguments)

def construct_image_to_image_pipeline_from_config(config_path: str) -> ImageToImagePipeline:
    with open(config_path, 'r') as file:
        data: dict = yaml.safe_load(file)["pipeline"]
        pipeline_args = {}
        for arg_name in data.keys():
            arg_data = data[arg_name]
            if arg_data["class"] == "Default":
                continue

            arg_args = arg_data["args"]
            if arg_args is None:
                arg_args = {}

            pipeline_args[arg_name] = construct_plugin_by_name(arg_data["class"],arg_args)
        
        return ImageToImagePipeline(**pipeline_args)