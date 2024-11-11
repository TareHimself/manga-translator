# This file was created using https://github.com/TareHimself/python-extractor created by https://github.com/TareHimself

import cv2
import math
import numpy as np
from typing import Union, Callable
from PIL import Image, ImageDraw, ImageFont
import pyphen
from collections import deque



##########################################################
# FILE => D:\Github\manga-translator\translator\utils.py #
##########################################################


def adjust_contrast_brightness(
    img: np.ndarray, contrast: float = 1.0, brightness: int = 0
):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)
def ensure_gray(img: np.ndarray):
    if len(img.shape) > 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    return img.copy()
def apply_mask(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray, inv=False):
    mask = ensure_gray(mask)
    a_loc, b_loc = foreground.copy(), background.copy()
    mask_inv = cv2.bitwise_not(mask)

    if inv:
        temp = mask
        mask = mask_inv
        mask_inv = temp

    a_loc = cv2.bitwise_and(a_loc, a_loc, mask=mask)
    b_loc = cv2.bitwise_and(b_loc, b_loc, mask=mask_inv)
    return cv2.add(a_loc, b_loc)
def cv2_to_pil(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil_to_cv2(img: Image) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def try_merge_hyphenated(text: list[str], max_chars: int):
    final = []
    total = deque(text)
    current = total.popleft().strip()

    while len(total) > 0 or current != "":
        if (
            len(total) > 0
            and current.endswith("-")
            and len(current[:-1] + total[0]) <= max_chars
        ):
            current = current[:-1] + total.popleft().strip()

        else:
            final.append(current)
            current = total.popleft() if len(total) > 0 else ""

    return final
def wrap_text(text: str, max_chars: int, hyphenator: Union[pyphen.Pyphen, None]):
    total = deque(list(filter(lambda a: len(a.strip()) > 0, text.split(" "))))
    current_word = total.popleft()
    lines = []
    current_line = ""
    while len(total) > 0 or len(current_word) > 0:
        sep = " " if len(current_line) > 0 else ""
        new_current = current_line + sep + current_word
        if len(new_current) > max_chars:
            space_left = max_chars - len(current_line + sep)

            try:
                if "-" in current_word:
                    idx = current_word.index("-")
                    total.appendleft(current_word[idx + 1 :])
                    current_word = current_word[:idx]
                    continue
                elif hyphenator is not None:
                    pairs = list(hyphenator.iterate(current_word))
                else:
                    pairs = []
            except:
                print("EXCEPTION WHEN HYPHENATING:", current_word)
                pairs = []
            if len(pairs) == 0:
                if current_line == "" and len(current_word) > max_chars:
                    return None
                lines.append(current_line)
                current_line = ""
                continue

            pair = min(pairs, key=lambda a: len(current_line + sep + a[0] + "-"))
            if len(current_line + sep + pair[0] + "-") > space_left:
                lines.append(current_line)
                if len(pair[0] + "-") <= max_chars:
                    lines.append(pair[0] + "-")
                    current_line = ""
                    current_word = pair[1]
                    continue
                else:
                    return None

            lines.append(current_line + sep + pair[0] + "-")
            current_line = ""
            current_word = pair[1]
        elif len(total) == 0:
            lines.append(current_line + sep + current_word)
            current_word = ""
        else:
            current_line = new_current
            current_word = total.popleft() if len(total) else ""

    return try_merge_hyphenated(lines, max_chars)
def get_average_font_size(font: ImageFont, text="some text here"):
    x, y, w, h = font.getbbox(text)
    widths = list(map(lambda a: font.getbbox(a)[2], list(text)))
    widths.sort(reverse=True)
    return widths[1] if len(widths) > 1 else widths[0], h
def get_best_font_size(
    text: str,
    wh: tuple[int, int],
    font_file: str,
    space_between_lines: int = 1,
    start_size: int = 18,
    step: int = 1,
    min_chars_per_line: int = 6,
    initial_iterations: int = 0,
    hyphenator: Union[pyphen.Pyphen, None] = None,
) -> Union[tuple[None, None, None, int], tuple[int, int, int, int]]:
    current_font_size = start_size
    current_font = None
    max_width, max_height = wh

    iterations = initial_iterations
    while True:
        iterations += 1

        if current_font_size < 0:
            return None, None, None, iterations

        current_font = ImageFont.truetype(font_file, current_font_size)

        cur_f_width, cur_f_height = get_average_font_size(current_font, text)

        chars_per_line = math.floor(max_width / cur_f_width)

        if chars_per_line < min_chars_per_line:
            current_font_size -= step
            continue

        # print(chars_per_line)
        lines = wrap_text(text, chars_per_line, hyphenator=hyphenator)
        if lines is None:
            current_font_size -= step
            continue

        height_needed = (len(lines) * cur_f_height) + (
            (len(lines) - 1) * space_between_lines
        )
        if height_needed <= max_height:
            return current_font_size, chars_per_line, cur_f_height, iterations
        current_font_size -= step
def draw_text_in_bubble(
    frame: np.ndarray,
    bounds: tuple[int],
    text="",
    font_file="fonts/animeace2_reg.ttf",
    color=(0, 0, 0),
    outline=1,
    outline_color: Union[tuple[int, int, int], None] = (255, 255, 255),
    hyphenator: Union[pyphen.Pyphen, None] = pyphen.Pyphen(lang="en"),
    offset: tuple[int, int] = (0, 0),
    rotation: int = 0,
):
    # print(color_diff(np.array(color),np.array((0,0,0))))
    if len(text) > 0:
        pt1, pt2 = bounds

        max_height = pt2[1] - pt1[1]
        max_width = pt2[0] - pt1[0]

        # fill background incase of segmentation errors
        # cv2.rectangle(frame, pt1, pt2, (255, 255, 255), -1)
        # cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 1)

        space_between_lines = 2
        font_size, chars_per_line, line_height, iters = get_best_font_size(
            text,
            (max_width, max_height),
            font_file,
            space_between_lines,
            30,
            1,
            hyphenator=hyphenator,
        )

        # frame = cv2.putText(
        #     frame,
        #     f"{font_size}",
        #     (0, 20),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.2,
        #     (255, 0, 0),
        #     1,
        #     cv2.LINE_AA,
        # )

        # print(text, font_size, chars_per_line, line_height, iters)
        if not font_size:
            return frame
        frame_as_pil = cv2_to_pil(frame)

        font = ImageFont.truetype(font_file, font_size)

        draw_x = pt1[0] + offset[0]
        draw_y = pt1[1] + offset[1]

        wrapped = wrap_text(text, chars_per_line, hyphenator=hyphenator)

        if rotation == 0:
            draw = ImageDraw.Draw(frame_as_pil)

            for line_no in range(len(wrapped)):
                line = wrapped[line_no]
                x, y, w, h = font.getbbox(line)
                draw.text(
                    (
                        draw_x + abs(((max_width - w) / 2)),
                        draw_y
                        + space_between_lines
                        + (
                            (
                                max_height
                                - (
                                    (len(wrapped) * line_height)
                                    + (len(wrapped) * space_between_lines)
                                )
                            )
                            / 2
                        )
                        + (line_no * line_height)
                        + (space_between_lines * line_no),
                    ),
                    str(line),
                    fill=(*color, 255),
                    font=font,
                    stroke_width=outline if outline_color is not None else 0,
                    stroke_fill=outline_color,
                )

            return pil_to_cv2(frame_as_pil)

        else:
            mask = frame_as_pil.copy()
            mask.paste((0, 0, 0), (0, 0, mask.size[0], mask.size[1]))

            draw = ImageDraw.Draw(frame_as_pil)

            draw_mask = ImageDraw.Draw(mask)

            for line_no in range(len(wrapped)):
                line = wrapped[line_no]
                x, y, w, h = font.getbbox(line)
                draw.text(
                    (
                        draw_x + abs(((max_width - w) / 2)),
                        draw_y
                        + space_between_lines
                        + (
                            (
                                max_height
                                - (
                                    (len(wrapped) * line_height)
                                    + (len(wrapped) * space_between_lines)
                                )
                            )
                            / 2
                        )
                        + (line_no * line_height)
                        + (space_between_lines * line_no),
                    ),
                    str(line),
                    fill=(*color, 255),
                    font=font,
                    stroke_width=outline if outline_color is not None else 0,
                    stroke_fill=outline_color,
                )

                draw_mask.text(
                    (
                        draw_x + abs(((max_width - w) / 2)),
                        draw_y
                        + space_between_lines
                        + (
                            (
                                max_height
                                - (
                                    (len(wrapped) * line_height)
                                    + (len(wrapped) * space_between_lines)
                                )
                            )
                            / 2
                        )
                        + (line_no * line_height)
                        + (space_between_lines * line_no),
                    ),
                    str(line),
                    fill=(255, 255, 255, 255),
                    font=font,
                    stroke_width=outline if outline_color is not None else 0,
                    stroke_fill=(255, 255, 255),
                )

            rotated_mask, rotated_frame = pil_to_cv2(mask.rotate(rotation)), pil_to_cv2(
                frame_as_pil.rotate(rotation)
            )

            final_mask_dilation = 3
            kernel = np.ones((final_mask_dilation, final_mask_dilation), np.uint8)
            rotated_mask = cv2.dilate(rotated_mask, kernel, iterations=1)
            rotated_mask = adjust_contrast_brightness(rotated_mask, 50, 200)

        return apply_mask(rotated_frame, frame, rotated_mask)
    else:
        return frame