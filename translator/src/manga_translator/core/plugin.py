import numpy as np
import torch

from manga_translator.core.typing import Vector4i, Vector2
from manga_translator.utils import get_available_pytorch_devices, get_languages
from typing import Any, Type, Union
#from translator.utils import run_in_thread_decorator


class Point:
    def __init__(self,x: int,y: int):
        self.x = x
        self.y = y

class Rect:
    def __init__(self,x: int,y: int,width: int,height: int):
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
    def __init__(self, id: str, name: str, description: str, default,convert_fn = None) -> None:
        self.id = id
        self.name = name
        self.type = PluginArgumentType.STRING
        self.description = description
        self.default = default
        self.convert_fn = convert_fn

    def get(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default": self.default,
            "type": self.type,
        }


class StringPluginArgument(PluginArgument):
    def __init__(self, id: str, name: str, description: str, default: str = "",convert_fn = None) -> None:
        super().__init__(id, name, description, default,convert_fn)
        self.type = PluginArgumentType.STRING

class IntPluginArgument(PluginArgument):
    def __init__(self, id: str, name: str, description: str, default: int = 0,convert_fn = None) -> None:
        super().__init__(id, name, description, default,convert_fn)
        self.type = PluginArgumentType.INT

class BooleanPluginArgument(PluginArgument):
    def __init__(self, id: str, name: str, description: str, default: bool = False,convert_fn = None) -> None:
        super().__init__(id, name, description, default,convert_fn)
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


class PluginSelectArgument(PluginArgument):
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        options: list[PluginSelectArgumentOption],
        default: str = "",
        convert_fn = None
    ) -> None:
        super().__init__(id, name, description, default,convert_fn)
        self.type = PluginArgumentType.SELECT
        self.options = options

    def get(self) -> dict[str, str]:
        data = super().get()
        data["options"] = [x.get() for x in self.options]
        return data
    

class LanguagePluginSelectArgument(PluginSelectArgument):
    language_options = list(map(lambda a: PluginSelectArgumentOption(a[1], a[0]), get_languages()))
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        default: str = "en",
    ) -> None:
        super().__init__(id, name, description,LanguagePluginSelectArgument.language_options ,default)

class PytorchDevicePluginSelectArgument(PluginSelectArgument):
    available_devices = get_available_pytorch_devices()
    available_devices_flat = list(map(lambda a: a[0],available_devices))
    available_devices_options = list(map(lambda a: PluginSelectArgumentOption(a[1],a[0]),available_devices))
    def __init__(
        self,
        id: str,
        name: str,
        description: str = "The pytorch device to use",
        default: str = (available_devices[1] if len(available_devices) > 0 else available_devices[0])[0],
    ) -> None:
        super().__init__(id, name, description,PytorchDevicePluginSelectArgument.available_devices_options,default,self.convert_to_torch_device)

    def convert_to_torch_device(self,device: str):
        if device == "cuda" or device == "cuda:0":
            return torch.device("cuda")
        
        if device not in PytorchDevicePluginSelectArgument.available_devices_flat:
            device = PytorchDevicePluginSelectArgument.available_devices_flat[0]
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
    def __init__(self, text: str = "", language: str = "en") -> None:
        self.text = text
        self.language = language


class OCR(BasePlugin):
    """Always outputs \"\" """

    def __init__(self) -> None:
        super().__init__()

    async def __call__(self, batch: list[np.ndarray]) -> list[OcrResult]:
        return await self.extract(batch)

    async def extract(self, batch: list[np.ndarray]):
        return [OcrResult("") for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Base Ocr"


class TranslatorResult:
    def __init__(self, text: str = "", lang_code: str = "en") -> None:
        self.lang_code = lang_code
        self.text = text


class Translator(BasePlugin):
    """Base Class for all Translator classes"""

    def __init__(self) -> None:
        super().__init__()

    async def __call__(self, batch: list[OcrResult]) -> list[TranslatorResult]:
        return await self.translate(batch)

    async def translate(self, batch: list[OcrResult]) -> list[TranslatorResult]:
        return [TranslatorResult(x.text) for x in batch]

    @staticmethod
    def get_name() -> str:
        return "Base Translator"

class DrawerInput:
    def __init__(self,draw_area: np.ndarray,translation: TranslatorResult) -> None:
        self.draw_area = draw_area
        self.translation = translation
        

class Drawer(BasePlugin):
    def __init__(self) -> None:
        super().__init__()


    async def draw(
        self, 
        frames: list[np.ndarray],
        translations: list[TranslatorResult]
    ) -> list[tuple[np.ndarray,np.ndarray]]:
        return [x.copy() for x in frames]

    async def __call__(
        self, 
        frames: list[np.ndarray],
        translations: list[TranslatorResult]
    ) -> tuple[list[np.ndarray],list[np.ndarray]]:
        return await self.draw(frames,translations)

class ColorDetector(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def detect_color(
        self,
        frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        return [np.array([0,0,0]) for _ in frames]

    async def __call__(
        self,
        frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        return await self.detect_color(frames)
    
class DetectionResult:
    def __init__(self,cls: int,rect: Vector4i,confidence: float) -> None:
        self.cls = cls
        self.bbox = rect
        self.confidence = confidence
    
class Detector(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def detect(
        self,
        frames: list[np.ndarray]
    ) -> list[list[DetectionResult]]:
        return [[] for _ in frames]

    async def __call__(
        self,
        frames: list[np.ndarray]
    ) -> list[list[DetectionResult]]:
        return await self.detect(frames)
    
class SegmentationResult:
    def __init__(self,cls: int,points: list[Vector2],confidence: float):
        self.cls = cls
        self.points = points
        self.confidence = confidence

class Segmenter(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def segment(
        self,
        frames: list[np.ndarray]
    ) -> list[list[SegmentationResult]]:
        return [[] for _ in frames]

    async def __call__(
        self,
        frames: list[np.ndarray]
    ) -> list[list[SegmentationResult]]:
        return await self.segment(frames)

class Cleaner(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def clean(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]] = [],
        detections: list[list[DetectionResult]] = []
    ) -> list[np.ndarray]:
        return [frame.copy() for frame in frames]

    async def __call__(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]] = [],
        detections: list[list[DetectionResult]] = []
    ) -> list[np.ndarray]:
        return await self.clean(frames,masks,segments,detections)
    
def construct_plugin(cls: Type[BasePlugin],arguments: dict[str,Any]):
    final_args = {}
    for arg in cls.get_arguments():
        if arg.convert_fn is not None:
            final_args[arg.id] = arg.convert_fn(arguments[arg.id])
        elif arg.id in arguments:
            final_args[arg.id] = arguments[arg.id]

    return cls(**final_args)