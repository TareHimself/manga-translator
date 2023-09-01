from typing import Any
import numpy as np
from PIL import ImageFont, ImageDraw
from numpy import ndarray
from hyphen import Hyphenator
from translator.core.plugin import BasePlugin, PluginArgument,PluginSelectArgument,PluginSelectArgumentOption
from translator.core.translators import TranslatorResult
from translator.utils import get_best_font_size, cv2_to_pil, pil_to_cv2, wrap_text,get_fonts



class Drawer(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    def draw(self,draw_color:np.ndarray,translation: TranslatorResult,frame: np.ndarray) -> np.ndarray:
        return frame
    
    def __call__(self, draw_color:np.ndarray,translation: TranslatorResult,frame: np.ndarray) -> np.ndarray:
        return self.draw(draw_color=draw_color,translation=translation,frame=frame)
    


class HorizontalDrawer(Drawer):
    """Draws text horizontaly"""

    def __init__(self,font_file="fonts/animeace2_reg.ttf",max_font_size = "30",line_spacing = "2") -> None:
        super().__init__()
        self.font_file = font_file
        self.max_font_size = round(float(max_font_size))
        self.line_spacing = round(float(line_spacing))

    def draw(self, draw_color: np.ndarray, translation: TranslatorResult, frame: np.ndarray) -> ndarray:
        # print(color_diff(np.array(color),np.array((0,0,0))))
        if len(translation.text.strip()) <= 0:
            return frame
    
        frame_h, frame_w, _ = frame.shape

        # fill background incase of segmentation errors
        # cv2.rectangle(frame, pt1, pt2, (255, 255, 255), -1)
        # cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 1)

        hyphenator = Hyphenator("en_US")

        font_size, chars_per_line, line_height, iters = get_best_font_size(
            translation.text,
            (frame_w, frame_h),
            font_file = self.font_file,
            space_between_lines= self.line_spacing,
            start_size= self.max_font_size,
            step = 1,
            hyphenator= hyphenator
        )

        if not font_size:
            return frame
        
        frame_as_pil = cv2_to_pil(frame)

        font = ImageFont.truetype(self.font_file, font_size)

        draw_x = 0
        draw_y = 0

        wrapped = wrap_text(translation.text, chars_per_line, hyphenator=hyphenator)
            
        image_draw = ImageDraw.Draw(frame_as_pil)

        for line_no in range(len(wrapped)):
            line = wrapped[line_no]
            x, y, w, h = font.getbbox(line)
            
            image_draw.text(
                (
                    draw_x + abs(((frame_w - w) / 2)),
                    draw_y
                    + self.line_spacing
                    + (
                            (
                                    frame_h
                                    - (
                                            (len(wrapped) * line_height)
                                            + (len(wrapped) * self.line_spacing)
                                    )
                            )
                            / 2
                    )
                    + (line_no * line_height)
                    + (self.line_spacing * line_no),
                ),
                str(line),
                fill=(*draw_color, 255),
                font=font,
                # stroke_width=outline if outline_color is not None else 0,
                # stroke_fill=outline_color
            )

        return pil_to_cv2(frame_as_pil)
    
    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        fonts_available = get_fonts()
        return [
            PluginSelectArgument(id="font_file",
                                 name="Font",
                                 description="The font to draw with",
                                 options=list(map(lambda a: PluginSelectArgumentOption(name=a[0],value=a[1]),fonts_available)),
                                 default=fonts_available[0][1]),
            PluginArgument(id="max_font_size",name="Max Font Size",description="The max font size for the sizing algorithm",default="30"),
            PluginArgument(id="line_spacing",name="Line Spacing",description="Space between lines",default="2")
        ]

    @staticmethod
    def get_name() -> str:
        return "Horizontal Drawer"
    
class VerticalDrawer(Drawer):
    """Draws text vertically"""
    
    def draw(self, draw_color: ndarray, translation: TranslatorResult, frame: ndarray) -> ndarray:
        return super().draw(draw_color, translation, frame)
    
    @staticmethod
    def is_valid() -> bool:
        return False
    
    @staticmethod
    def get_name() -> str:
        return "Vertical Drawer"
    


def get_drawers() -> list[Drawer]:
    return list(filter(lambda a: a.is_valid(), [HorizontalDrawer,VerticalDrawer]))
    
