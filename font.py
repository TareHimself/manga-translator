from utils import cv2_to_pil, pil_to_cv2, pixels_to_pt, pt_to_pixels
import cv2
import numpy as np
import textwrap
from PIL import ImageDraw, ImageFont
import math

# frame_to_pil = cv2_to_pil(frame)
# draw = ImageDraw.Draw(frame_to_pil)


# font = ImageFont.truetype("fonts/BlambotClassicBB.ttf", 5)
# draw_x = pt1[0] + (max_width / 2)
# draw_y = pt1[1]
# approx_h = 0
# for line_no in range(len(wrapped)):
#     line = wrapped[line_no]
#     text_w, text_h = font.getsize(line)
#     approx_h = text_h if approx_h == 0 else approx_h
#     # print(approx_h, line_no * approx_h)
#     draw.text(
#         (draw_x - (text_w / 2), draw_y + (line_no * approx_h)),
#         str(line),
#         fill=(0, 0, 0, 255),
#         font=font,
#     )

# pil_to_cv2(frame_to_pil)

font = ImageFont.truetype("fonts/BlambotClassicBB.ttf", 18)


def get_average_font_size(font, text="some text here"):
    x, y, w, h = font.getbbox(text)
    return round((w / len(text))), h


def get_best_font_size(
    text, wh, font_file, start_size=18, step=1
):
    current_font_size = start_size
    current_font = None

    max_width, max_height = wh

    iterations = 0
    while True:
        iterations += 1
        current_font = ImageFont.truetype(font_file, current_font_size)
        if current_font_size < 0:
            return None, None, None, iterations
        cur_f_width, cur_f_height = get_average_font_size(current_font, text)
        chars_per_line = math.floor(max_width / cur_f_width)
        if chars_per_line == 0:
            current_font_size -= step
            continue
        lines = textwrap.wrap(text, chars_per_line)
        height_needed = (len(lines) * cur_f_height)
        if height_needed < max_height:
            return current_font_size, chars_per_line, cur_f_height, iterations
        current_font_size -= step


text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
test_image = np.zeros((200, 200, 3), dtype=np.uint8)

frame_to_pil = cv2_to_pil(test_image)
draw = ImageDraw.Draw(frame_to_pil)

font_size, max_line_width, line_height, iters = get_best_font_size(
    text,
    (test_image.shape[0], test_image.shape[1]),
    "fonts/BlambotClassicBB.ttf",
    30,
    1,
)

font = ImageFont.truetype("fonts/BlambotClassicBB.ttf", font_size)
draw_x = 0
draw_y = 0
wrapped = textwrap.wrap(text, max_line_width)
for line_no in range(len(wrapped)):
    line = wrapped[line_no]
    draw.text(
        (draw_x, draw_y + (line_no * line_height)),
        str(line),
        fill=(255, 255, 255, 255),
        font=font,
    )

final_img = pil_to_cv2(frame_to_pil)

cv2.imshow("Final", final_img)
cv2.waitKey(0)
