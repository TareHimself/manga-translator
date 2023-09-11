from typing import Any
from numpy import ndarray
from translator.core.plugin import Drawer, TranslatorResult
from translator.utils import (
    get_best_font_size,
    cv2_to_pil,
    pil_to_cv2,
    wrap_text,
    get_fonts,
)


class VerticalDrawer(Drawer):
    """Draws text vertically"""

    async def draw(
        self, draw_color: ndarray, translation: TranslatorResult, frame: ndarray
    ) -> ndarray:
        return super().draw(draw_color, translation, frame)

    @staticmethod
    def is_valid() -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "Vertical Drawer"
