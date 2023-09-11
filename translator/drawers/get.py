from translator.core.plugin import Drawer
from translator.drawers.horizontal import HorizontalDrawer
from translator.drawers.vertical import VerticalDrawer


def get_drawers() -> list[Drawer]:
    return list(filter(lambda a: a.is_valid(), [HorizontalDrawer, VerticalDrawer]))
