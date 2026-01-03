from manga_translator.core.plugin import Drawer
from manga_translator.drawing.horizontal import HorizontalDrawer

_drawing_data = list(
    filter(
        lambda a: a.is_valid(),
        [HorizontalDrawer],
    )
)


def get_drawers() -> list[Drawer]:
    return _drawing_data
