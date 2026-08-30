from typing import Optional
import numpy as np
from PIL import ImageDraw
import pyphen
import asyncio
from comic_localizer.core.plugin import (
    ColorDetectionResult,
    Drawer,
    StringPluginArgument,
    PluginArgument,
    IntPluginArgument,
    TranslatorResult,
)
from comic_localizer.utils import (
    find_best_font_size,
    cv2_to_pil,
    load_font,
    pil_to_cv2,
    ensure_gray,
)


class HorizontalDrawer(Drawer):
    """Draws text horizontaly"""

    def __init__(
        self,
        font_file: Optional[str] = None,  # required; see fonts/FONTS.md
        max_font_size=20,
        min_font_size=5,
        line_spacing=2,
        hyphenator: Optional[pyphen.Pyphen] = pyphen.Pyphen(lang="en"),
        margin=3,
    ) -> None:
        super().__init__()
        if not font_file:
            raise ValueError("HorizontalDrawer needs a font_file (see fonts/FONTS.md)")
        self.font_file = font_file
        self.max_font_size = round(max_font_size)
        self.min_font_size = round(min_font_size)
        self.line_spacing = round(line_spacing)
        self.hyphenator = hyphenator
        self.margin = margin

    # TODO make this whole function better
    def draw_text(
        self,
        frame: np.ndarray,
        translation: TranslatorResult,
        color: ColorDetectionResult,
    ):

        frame_h, frame_w = frame.shape[:2]

        if len(translation.text.strip()) <= 0:
            return frame.copy(), np.full((frame_h, frame_w), 0, dtype=np.uint8)

        fit_result = find_best_font_size(
            translation.text,
            self.font_file,
            (frame_w - (self.margin * 2), frame_h - (self.margin * 2)),
            20,
            max_font_size=self.max_font_size,
            min_font_size=self.min_font_size,
            line_spacing=self.line_spacing,
            hyphenator=self.hyphenator,
            outline_size=color.outline_size,
        )

        if fit_result is None:
            # cv2.imshow("Fit fail",frame)
            # cv2.waitKey(0)
            return frame.copy(), np.full((frame_h, frame_w), 0, dtype=np.uint8)

        mask = np.full_like(frame, 0, dtype=np.uint8)
        as_pil = cv2_to_pil(frame)
        mask_as_pill = cv2_to_pil(mask)

        pen = ImageDraw.Draw(as_pil)

        mask_pen = ImageDraw.Draw(mask_as_pill)

        font = load_font(self.font_file, size=fit_result.font_size)
        text_bounds = np.array(fit_result.wrap.bounds)
        available_space_y = frame_h
        centering_offset_y = (available_space_y - text_bounds[1]) / 2

        for i in range(len(fit_result.wrap.lines)):
            line = fit_result.wrap.lines[i]

            text = " ".join(line.words)

            x1, y1, x2, y2 = font.getbbox(text)

            w = x2 + x1

            centering_offset_x = (frame_w - w) / 2
            x_pos = centering_offset_x
            y_pos = centering_offset_y + line.offset

            pen.text(
                (
                    x_pos,
                    y_pos,
                ),
                str(text),
                fill=(*color.text_color, 255),
                font=font,
                stroke_width=color.outline_size,
                stroke_fill=(
                    (*color.outline_color, 255) if color.outline_size > 0 else None
                ),
            )

            # text mask for compositing
            mask_pen.text(
                (
                    x_pos,
                    y_pos,
                ),
                str(text),
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=color.outline_size,
                stroke_fill=(255, 255, 255, 255),
            )

        return pil_to_cv2(as_pil), ensure_gray(pil_to_cv2(mask_as_pill))

    async def draw(
        self,
        frames: list[np.ndarray],
        translations: list[TranslatorResult],
        colors: list[ColorDetectionResult],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return await asyncio.gather(
            *[
                asyncio.to_thread(self.draw_text, frame, translation, color)
                for frame, translation, color in zip(frames, translations, colors)
            ]
        )

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="font_file",
                name="Font File",
                description="The font file to draw with",
                default="",
            ),
            IntPluginArgument(
                id="max_font_size",
                name="Max Font Size",
                description="The max font size for the sizing algorithm",
                default=30,
            ),
            IntPluginArgument(
                id="min_font_size",
                name="Min Font Size",
                description="The min font size for the sizing algorithm",
                default=5,
            ),
            IntPluginArgument(
                id="line_spacing",
                name="Line Spacing",
                description="Space between lines",
                default=2,
            ),
            IntPluginArgument(
                id="margin",
                name="Margin",
                description="",
                default=3,
            ),
        ]

    @staticmethod
    def get_name() -> str:
        return "Horizontal"
