from typing import Any
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw
from numpy import ndarray
from hyphen import Hyphenator
import asyncio
from translator.core.plugin import (
    Drawable,
    Drawer,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
    TranslatorResult,
)
from translator.utils import (
    get_best_font_size,
    cv2_to_pil,
    pil_to_cv2,
    wrap_text,
    get_fonts,
    display_image,
    TranslatorGlobals
)
from translator.color_detect.utils import luminance_similarity


class HorizontalDrawer(Drawer):
    """Draws text horizontaly"""

    def __init__(
        self, font_file="fonts/animeace2_reg.ttf", max_font_size="30", line_spacing="2"
    ) -> None:
        super().__init__()
        self.font_file = font_file
        self.max_font_size = round(float(max_font_size))
        self.line_spacing = round(float(line_spacing))

    async def draw(
        self,batch: list[Drawable]
    ) -> list[tuple[ndarray,ndarray]]:
        return await asyncio.gather(*[self.draw_one(x) for x in batch])
                
    
    async def draw_one(
        self, item: Drawable
    ) -> tuple[ndarray,ndarray]:
        item_mask = np.zeros_like(item.frame)
        if len(item.translation.text.strip()) <= 0:
            return (item.frame,item_mask)

        frame_h, frame_w, _ = item.frame.shape

        # fill background incase of segmentation errors
        # cv2.rectangle(frame, pt1, pt2, (255, 255, 255), -1)
        # cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 1)

        hyphenator = Hyphenator("en_US")

        font_size, chars_per_line, line_height, iters = get_best_font_size(
            item.translation.text,
            (frame_w, frame_h),
            font_file=self.font_file,
            space_between_lines=self.line_spacing,
            start_size=self.max_font_size,
            step=1,
            hyphenator=hyphenator,
        )

        if not font_size:
            return (item.frame,item_mask)

        font = ImageFont.truetype(self.font_file, font_size)

        font
        draw_x = 0
        draw_y = 0

        wrapped = wrap_text(item.translation.text, chars_per_line, hyphenator=hyphenator)

        frame_as_pil = cv2_to_pil(item.frame)
        
        mask_as_pil = cv2_to_pil(item_mask)

        image_draw = ImageDraw.Draw(frame_as_pil)

        mask_draw = ImageDraw.Draw(mask_as_pil)
        # color_fg = item.color
        # avg_frame_color = np.mean(item.frame, axis=(0, 1))
        # frame_to_text_sim = luminance_similarity(color_fg,avg_frame_color)
        # color_bg = np.array([255,255,255]).astype(np.uint8)

        # if frame_to_text_sim > 0.5:
        #     stroke_width = 2
        #     sim_to_white = luminance_similarity(color_fg,TranslatorGlobals.COLOR_WHITE)
        #     sim_to_black = luminance_similarity(color_fg,TranslatorGlobals.COLOR_BLACK)
        #     if sim_to_black < sim_to_white:
        #         color_bg = np.array([0,0,0]).astype(np.uint8)

        color_fg,color_bg,should_do_bg = item.color
        
        stroke_width = 2 if should_do_bg else 0

        # print("SIMILARITY",luminance_similarity(item.color[0],item.color[1]),item.color)
        # print("DRAWING",item.translation.text)
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
                fill=(*color_fg,255),
                font=font,
                stroke_width=stroke_width,
                stroke_fill=(*color_bg,255) if stroke_width > 0 else None
            )

            mask_draw.text(
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
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=stroke_width,
                stroke_fill=(255, 255, 255) if stroke_width > 0 else None
            )

        mask_cv2 = cv2.cvtColor(pil_to_cv2(mask_as_pil),cv2.COLOR_BGR2GRAY)

        _, binary_mask = cv2.threshold(mask_cv2, 1, 255, cv2.THRESH_BINARY)

        return (pil_to_cv2(frame_as_pil),binary_mask)
        

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        fonts_available = get_fonts()
        return [
            PluginSelectArgument(
                id="font_file",
                name="Font",
                description="The font to draw with",
                options=list(
                    map(
                        lambda a: PluginSelectArgumentOption(name=a[0], value=a[1]),
                        fonts_available,
                    )
                ),
                default=fonts_available[0][1],
            ),
            PluginArgument(
                id="max_font_size",
                name="Max Font Size",
                description="The max font size for the sizing algorithm",
                default="30",
            ),
            PluginArgument(
                id="line_spacing",
                name="Line Spacing",
                description="Space between lines",
                default="2",
            ),
        ]

    @staticmethod
    def get_name() -> str:
        return "Horizontal Drawer"
