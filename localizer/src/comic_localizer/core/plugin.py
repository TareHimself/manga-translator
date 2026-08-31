import numpy as np
import torch
from comic_localizer.core.typing import Vector4i, Vector3u8
from comic_localizer.utils import (
    get_available_pytorch_devices,
    get_default_language,
    standardize_language_code,
    inverse_luminance_color,
    perf_async,
)
from typing import Any, Type, Optional
from .constants import DetectionType, SegmentationType

# from translator.utils import run_in_thread_decorator


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Rect:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class PluginArgumentType:
    STRING = 0
    SELECT = 1
    INT = 2
    BOOLEAN = 3


class PluginArgument:
    def __init__(
        self, id: str, name: str, description: str, default, convert_fn=None
    ) -> None:
        self.id = id
        self.name = name
        self.type = PluginArgumentType.STRING
        self.description = description
        self.default = default
        self.convert_fn = convert_fn

    def get(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default": self.default,
            "type": self.type,
        }


class StringPluginArgument(PluginArgument):
    def __init__(
        self, id: str, name: str, description: str, default: str = "", convert_fn=None
    ) -> None:
        super().__init__(id, name, description, default, convert_fn)
        self.type = PluginArgumentType.STRING


class IntPluginArgument(PluginArgument):
    def __init__(
        self, id: str, name: str, description: str, default: int = 0, convert_fn=None
    ) -> None:
        # clients may hand args in as strings (form posts, yaml); coerce
        super().__init__(id, name, description, default, convert_fn or int)
        self.type = PluginArgumentType.INT


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class BooleanPluginArgument(PluginArgument):
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        default: bool = False,
        convert_fn=None,
    ) -> None:
        super().__init__(id, name, description, default, convert_fn or _to_bool)
        self.type = PluginArgumentType.BOOLEAN


class PluginSelectArgumentOption:
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value

    def get(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value": self.value,
        }


class SelectPluginArgument(PluginArgument):
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        options: list[PluginSelectArgumentOption],
        default: str = "",
        convert_fn=None,
    ) -> None:
        super().__init__(id, name, description, default, convert_fn)
        self.type = PluginArgumentType.SELECT
        self.options = options

    def get(self) -> dict[str, str]:
        data = super().get()
        data["options"] = [x.get() for x in self.options]
        return data


class LanguageStringArgument(StringPluginArgument):
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        default: str = get_default_language(),
    ) -> None:
        super().__init__(id, name, description, default, self.convert_lang)

    def convert_lang(self, language_text: str) -> str:
        return standardize_language_code(language_text)


class PytorchDevicePluginArgument(SelectPluginArgument):
    available_devices = get_available_pytorch_devices()
    available_devices_flat = [a[0] for a in available_devices]
    available_devices_options = [
        PluginSelectArgumentOption(a[1], a[0]) for a in available_devices
    ]

    def __init__(
        self,
        id: str,
        name: str,
        description: str = "The pytorch device to use",
        default: str = (
            available_devices[1] if len(available_devices) > 1 else available_devices[0]
        )[0],
    ) -> None:
        super().__init__(
            id,
            name,
            description,
            PytorchDevicePluginArgument.available_devices_options,
            default,
            self.convert_to_torch_device,
        )

    def convert_to_torch_device(self, device: str):
        if device == "cuda" or device == "cuda:0":
            return torch.device("cuda")

        if device not in PytorchDevicePluginArgument.available_devices_flat:
            device = PytorchDevicePluginArgument.available_devices_flat[0]
        return torch.device(device)


class BasePlugin:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_name() -> str:
        return "unknown"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return []

    @staticmethod
    def is_valid() -> bool:
        return True


class OcrResult:
    def __init__(self, text: str = "", language: str = get_default_language()) -> None:
        self._language = language
        self.language = language
        self.text = text

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = standardize_language_code(value)


class OCR(BasePlugin):
    """Always outputs \"\""""

    def __init__(self) -> None:
        super().__init__()

    @perf_async(name_override="extract")
    async def __call__(self, batch: list[np.ndarray]) -> list[OcrResult]:
        return await self.extract(batch)

    async def extract(self, batch: list[np.ndarray]):
        """
        Extract text from the images, if an ocr result is empty it will be assumed that no text was in the frame
        :param self: Description
        :param batch: Description
        :type batch: list[np.ndarray]
        """
        return [OcrResult("") for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Base Ocr"


class TranslatorResult:
    def __init__(self, text: str = "", language: str = get_default_language()) -> None:
        self._language = language
        self.language = language
        self.text = text

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = standardize_language_code(value)


class Translator(BasePlugin):
    """Base Class for all Translator classes"""

    def __init__(self) -> None:
        super().__init__()

    @perf_async(name_override="translate")
    async def __call__(self, batch: list[OcrResult]) -> list[TranslatorResult]:
        return await self.translate(batch)

    async def translate(self, batch: list[OcrResult]) -> list[TranslatorResult]:
        return [TranslatorResult(x.text) for x in batch]

    @staticmethod
    def get_name() -> str:
        return "Base Translator"


class ColorDetectionResult:
    def __init__(
        self,
        text_color: Vector3u8,
        outline_size: int = 0,
        outline_color: Optional[Vector3u8] = None,
    ):
        self.text_color = text_color
        self.outline_size = outline_size
        self.outline_color = (
            inverse_luminance_color(self.text_color)
            if outline_color is None
            else outline_color
        )


class ColorDetector(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def detect_color(
        self,
        text: list[np.ndarray],
        cleaned: list[np.ndarray],
        original: list[np.ndarray],
    ) -> list[ColorDetectionResult]:
        return [ColorDetectionResult(np.zeros((3), dtype=np.uint8), 1) for _ in text]

    @perf_async(name_override="detect_color")
    async def __call__(
        self,
        text: list[np.ndarray],
        cleaned: list[np.ndarray],
        original: list[np.ndarray],
    ) -> list[ColorDetectionResult]:
        return await self.detect_color(text, cleaned, original)


class Drawer(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def draw(
        self,
        frames: list[np.ndarray],
        translations: list[TranslatorResult],
        colors: list[ColorDetectionResult],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(x.copy(), x.copy()) for x in frames]

    @perf_async(name_override="draw")
    async def __call__(
        self,
        frames: list[np.ndarray],
        translations: list[TranslatorResult],
        colors: list[ColorDetectionResult],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return await self.draw(frames, translations, colors)


class DetectionResult:
    def __init__(self, type: DetectionType, bbox: Vector4i, confidence: float) -> None:
        self.type = type
        self.bbox = bbox
        self.confidence = confidence


class Detector(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def detect(self, frames: list[np.ndarray]) -> list[list[DetectionResult]]:
        return [[] for _ in frames]

    @perf_async(name_override="detect")
    async def __call__(self, frames: list[np.ndarray]) -> list[list[DetectionResult]]:
        return await self.detect(frames)


class SegmentationResult:
    def __init__(
        self, type: SegmentationType, points: np.ndarray, confidence: float
    ):
        self.type = type
        self.points = points
        self.confidence = confidence


class Segmenter(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def segment(self, frames: list[np.ndarray]) -> list[list[SegmentationResult]]:
        return [[] for _ in frames]

    @perf_async(name_override="segment")
    async def __call__(
        self, frames: list[np.ndarray]
    ) -> list[list[SegmentationResult]]:
        return await self.segment(frames)


class Cleaner(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def clean(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]],
        detections: list[list[DetectionResult]],
    ) -> list[np.ndarray]:
        return [frame.copy() for frame in frames]

    @perf_async(name_override="clean")
    async def __call__(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]],
        detections: list[list[DetectionResult]],
    ) -> list[np.ndarray]:
        return await self.clean(frames, masks, segments, detections)


def construct_plugin(cls: Type[BasePlugin], arguments: dict[str, Any]):
    final_args = {}
    for arg in cls.get_arguments():
        if arg.id in arguments:
            if arg.convert_fn is not None:
                final_args[arg.id] = arg.convert_fn(arguments[arg.id])
            else:
                final_args[arg.id] = arguments[arg.id]

    return cls(**final_args)
