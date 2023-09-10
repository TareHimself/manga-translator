import numpy as np

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
    def __init__(self, id: str, name: str, description: str, options: list[PluginSelectArgumentOption],
                 default: str = "") -> None:
        super().__init__(id, name, description, default)
        self.type = PluginArgumentType.SELECT
        self.options = options

    def get(self) -> dict[str, str]:
        data = super().get()
        data['options'] = [x.get() for x in self.options]
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
    """Always outputs \"Sample\""""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text: np.ndarray) -> OcrResult:
        return self.do_ocr(text)

    def do_ocr(self, text: np.ndarray):
        return OcrResult("Sample")

    @staticmethod
    def get_name() -> str:
        return "Base Ocr"

class TranslatorResult():
    def __init__(self,text:str = '',lang_code: str = 'en') -> None:
        self.lang_code = lang_code
        self.text = text

class Translator(BasePlugin):
    """Base Class for all Translator classes"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, ocr_result: OcrResult) -> str:
        return self.translate(ocr_result)

    def translate(self, ocr_result: OcrResult) -> TranslatorResult:
        return TranslatorResult(ocr_result.text)

    @staticmethod
    def get_name() -> str:
        return "Base Translator"
    

class Drawer(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    def draw(self,draw_color:np.ndarray,translation: TranslatorResult,frame: np.ndarray) -> np.ndarray:
        return frame
    
    def __call__(self, draw_color:np.ndarray,translation: TranslatorResult,frame: np.ndarray) -> np.ndarray:
        return self.draw(draw_color=draw_color,translation=translation,frame=frame)
    

class Cleaner(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    def clean(self,frame: np.ndarray,mask: np.ndarray,detection_results: list[tuple[tuple[int,int,int,int],str,float]] = []) -> tuple[np.ndarray,np.ndarray]:
        return frame
    
    def __call__(self,frame: np.ndarray,mask: np.ndarray,detection_results: list[tuple[tuple[int,int,int,int],str,float]] = []) -> tuple[np.ndarray,np.ndarray]:
        return self.clean(frame=frame,mask=mask,detection_results=detection_results)