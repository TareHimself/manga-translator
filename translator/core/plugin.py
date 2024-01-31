import numpy as np
from translator.utils import run_in_thread_decorator

class PluginArgumentType:
    TEXT = 0
    SELECT = 1


class PluginArgument:
    def __init__(self, id: str, name: str, description: str, default: str = "") -> None:
        self.id = id
        self.name = name
        self.type = PluginArgumentType.TEXT
        self.description = description
        self.default = default

    def get(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default": self.default,
            "type": self.type,
        }


class PluginTextArgument(PluginArgument):
    def __init__(self, id: str, name: str, description: str, default: str = "") -> None:
        super().__init__(id, name, description, default)
        self.type = PluginArgumentType.TEXT


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
    ) -> None:
        super().__init__(id, name, description, default)
        self.type = PluginArgumentType.SELECT
        self.options = options

    def get(self) -> dict[str, str]:
        data = super().get()
        data["options"] = [x.get() for x in self.options]
        return data


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


class Ocr(BasePlugin):
    """Always outputs \"Sample\" """

    def __init__(self) -> None:
        super().__init__()

    async def __call__(self, batch: list[np.ndarray]) -> list[OcrResult]:
        return await self.do_ocr(batch)

    async def do_ocr(self, batch: list[np.ndarray]):
        return [OcrResult("Sample") for _ in batch]

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

class Drawable:
    def __init__(self,color: tuple[np.ndarray,np.ndarray,bool], translation: TranslatorResult,frame: np.ndarray) -> None:
        self.color = color
        self.translation = translation
        self.frame = frame

class Drawer(BasePlugin):
    def __init__(self) -> None:
        super().__init__()


    async def draw(
        self, batch: list[Drawable]
    ) -> list[tuple[np.ndarray,np.ndarray]]:
        return [x.frame for x in batch]

    async def __call__(
        self, batch: list[Drawable]
    ) -> list[tuple[np.ndarray,np.ndarray]]:
        return await self.draw(batch=batch)


class Cleaner(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def clean(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = [],
    ) -> tuple[np.ndarray, np.ndarray]:
        return frame

    async def __call__(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = [],
    ) -> tuple[np.ndarray, np.ndarray]:
        return await self.clean(frame=frame, mask=mask, detection_results=detection_results)
