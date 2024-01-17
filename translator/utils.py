import random
import cv2
import json
import os
import math
import time
import shutil
import torch
import threading
import pycountry
import numpy as np
import PIL
import PySimpleGUI as sg
import asyncio
import inspect
import largestinteriorrectangle as lir
from torchvision import transforms
from typing import Union, Callable
from PIL import Image, ImageDraw, ImageFont
from hyphen import Hyphenator
from tqdm import tqdm
from collections import deque
import traceback


class TranslatorGlobals:
    COLOR_BLACK = np.array((0, 0, 0))
    COLOR_WHITE = np.array((255, 255, 255))

async def run_in_thread(func,*args,**kwargs):
    loop = asyncio.get_event_loop()
    task = asyncio.Future()
    def run():
        nonlocal loop
        nonlocal func
        nonlocal task
        
        result = func(*args,**kwargs)
        
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        loop.call_soon_threadsafe(task.set_result,result)
    
    task_thread = threading.Thread(group=None,daemon=True,target=run)
    task_thread.start()
    return await task

def run_in_thread_decorator(func):
    async def wrapper(*args,**kwargs):
        return await run_in_thread(func,*args,**kwargs) # Comment this out to disable threading
    

        result = func(*args,**kwargs)
        if inspect.isawaitable(result):
                result = await result
        return result
    return wrapper


    
def get_torch_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

def simplify_lang_code(code: str) -> Union[str, None]:
    try:
        lang = pycountry.languages.lookup(code)

        return getattr(lang, "alpha_2", getattr(lang, "alpha_3", None))
    except:
        return code


def get_languages() -> list[tuple[str, str]]:
    return list(
        filter(
            lambda a: a[1] is not None,
            list(
                map(
                    lambda a: (
                        a.name,
                        getattr(a, "alpha_2", getattr(a, "alpha_3", None)),
                    ),
                    list(pycountry.languages),
                )
            ),
        )
    )


def lang_code_to_name(code: str) -> Union[str, None]:
    try:
        return pycountry.languages.lookup(code).name
    except:
        return None


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


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} Took {((time.time() - start) * 1000):.6f} MS")
        return result

    return wrapper


def has_white(image: np.ndarray):
    # Set RGB values for white
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)

    # Find white pixels within the specified range
    white_pixels = cv2.inRange(image, white_lower, white_upper)

    # Check if any white pixels were found
    return cv2.countNonZero(white_pixels) > 0


display_image_lock = threading.Lock()


def display_image(img: np.ndarray, name: str = "debug"):
    global display_image_lock

    with display_image_lock:
        # Convert the CV2 image array to a format compatible with PySimpleGUI
        image_bytes = cv2.imencode(".png", img)[1].tobytes()

        # Create the GUI layout
        layout = [
            [sg.Text(text=name)],
            [sg.Image(data=image_bytes)],
            [sg.Button("Save"), sg.Button("Close")],
        ]

        # Create the window
        window = sg.Window(name, layout)

        # Event loop to handle events
        while True:
            event, values = window.read()
            if event == sg.WINDOW_CLOSED or event == "Close":
                break

            if event == "Save":
                cv2.imwrite(name + ".png", img)

        # Close the window
        window.close()


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


def make_bubble_mask(frame: np.ndarray):
    image = frame.copy()
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use the Canny edge detection algorithm
    edges = cv2.Canny(blurred, 50, 150)

    # # Apply morphological closing to close gaps in the circle
    # kernel = np.ones((9, 9), np.uint8)
    # closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # _, binary_image = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Create a black image with the same size as the original

    stage_1 = cv2.drawContours(
        np.zeros_like(image), contours, -1, (255, 255, 255), thickness=2
    )

    stage_1 = cv2.bitwise_not(stage_1)

    stage_1 = cv2.cvtColor(stage_1, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(stage_1, 200, 255, cv2.THRESH_BINARY)

    # Find connected components in the binary image
    num_labels, labels = cv2.connectedComponents(binary_image)

    largest_island_label = np.argmax(np.bincount(labels.flat)[1:]) + 1

    mask = np.zeros_like(image)

    mask[labels == largest_island_label] = 255

    # mask = cv2.bitwise_not(mask)

    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove black spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return adjust_contrast_brightness(mask, 100)


def get_masked_bounds(mask: np.ndarray):
    gray = ensure_gray(mask)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 200, 255, 0)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    all_contours = []
    for c in contours:
        all_contours.extend(c)
    # display_image(mask,"Test")
    x, y, w, h = cv2.boundingRect(np.array(all_contours))

    return x, y, x + w, y + h


def get_histogram_for_region(frame: np.ndarray, region_mask=None):
    masked_frame = cv2.bitwise_and(
        frame,
        frame,
        mask=ensure_gray(
            region_mask if region_mask is not None else np.full_like(frame, 255)
        ),
    )
    return cv2.calcHist(
        [masked_frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )


def mask_text_and_make_bubble_mask(
    frame: np.ndarray, frame_text_mask: np.ndarray, frame_cleaned: np.ndarray
):
    # debug_image(frame_cleaned)
    x1, y1, x2, y2 = get_masked_bounds(frame_text_mask)

    frame_section = frame.copy()[y1:y2, x1:x2]

    mask_section = frame_text_mask.copy()[y1:y2, x1:x2]

    text = apply_mask(
        frame_section,
        np.full(frame_section.shape, 255, dtype=frame_section.dtype),
        mask_section,
    )

    return text, make_bubble_mask(frame_cleaned)


def cv2_to_pil(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img: Image) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_bounds_for_text(frame_mask: np.ndarray):
    gray = ensure_gray(frame_mask)
    # Threshold the image
    ret, thresh = cv2.threshold(gray, 200, 255, 0)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    largest_contour = max(contours, key=cv2.contourArea)
    polygon = np.array([largest_contour[:, 0, :]])

    rect = lir.lir(polygon)

    return lir.pt1(rect), lir.pt2(rect)


def fix_order_after_intersection_fix(
    a1: int, a2: int, b1: int, b2: int, was_sorted: bool, was_intersecting: bool = False
):
    if was_sorted:
        return b1, b2, a1, a2, was_intersecting

    return a1, a2, b1, b2, was_intersecting


def fix_intersection(
    a1: int, a2: int, b1: int, b2: int, sorted: bool = False, was_sorted: bool = False
):
    if not sorted:
        if a1[0] < b1[0]:
            return fix_intersection(a1, a2, b1, b2, True)
        else:
            return fix_intersection(b1, b2, a1, a2, True, True)

    a_width = a2[0] - a1[0]
    a_height = a2[1] - a1[1]
    b_width = b2[0] - b1[0]
    b_height = b2[1] - b1[1]
    # a is above b
    if a1[1] < b1[1]:
        is_intersecting = a1[0] < b1[0] < a2[0] and a1[1] < b1[1] < a2[1]
        if not is_intersecting:
            return fix_order_after_intersection_fix(a1, a2, b1, b2, was_sorted)

        midpoint_x = max(abs(int(((a2[0] - b1[0]) / 2))) + 1, 3)

        midpoint_y = abs(int(((a2[1] - b1[1]) / 2))) + 1

        # if a_height < b_height:
        #     b1[1] = a2[1]
        # else:
        #     a2[1] = b1[1]

        # b1[0] += midpoint_x
        # a2[0] -= midpoint_x
        return fix_order_after_intersection_fix(a1, a2, b1, b2, was_sorted, True)
    else:
        is_intersecting = a1[0] < b1[0] < a2[0] and b1[1] < a1[1] < b2[1]
        if not is_intersecting:
            return fix_order_after_intersection_fix(a1, a2, b1, b2, was_sorted)

        midpoint_x = max(abs(int(((a2[0] - b1[0]) / 2))) + 1, 3)

        midpoint_y = abs(int(((b2[1] - a1[0]) / 2))) + 1

        # if b_height < a_height:
        #     a1[1] = b2[1]
        # else:
        #     b2[1] = a1[1]
        # b1[0] += midpoint_x
        # a2[0] -= midpoint_x

        return fix_order_after_intersection_fix(a1, a2, b1, b2, was_sorted, True)


def simplify_points(points: np.ndarray):
    # Find the convex hull of the points
    hull = cv2.convexHull(points)

    # Convert the convex hull back to a list of points
    simplified_points = np.array([np.array(x) for x in hull.squeeze().tolist()])

    return simplified_points


def mask_text_for_in_painting(frame: np.ndarray, mask: np.ndarray):
    image = frame.copy()

    hist = get_histogram_for_region(frame, np.full_like(frame, 255, dtype=frame.dtype))

    # Find the bin with the highest frequency
    max_bin = np.unravel_index(hist.argmax(), hist.shape)

    # Retrieve the corresponding color value
    is_white = (
        ((max_bin[2] + max_bin[1] + max_bin[0]) / 3) / 255
    ) > 0.5  # checks if the dominant color is bright or dark with a 0.5 threshold

    if not is_white:
        image = cv2.bitwise_not(image)

    # Convert the image to grayscale
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
    )  # 15, 5)

    # Perform morphological operations to improve the text extraction
    kernel_size = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    opening = cv2.bitwise_and(opening, opening, mask=ensure_gray(mask))

    # Find contours of the characters
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image
    new_mask = np.zeros_like(image)

    # Draw contours on the mask
    for contour in contours:
        # Filter out small contours and contours with a large aspect ratio
        (x, y, w, h) = cv2.boundingRect(contour)
        ratio = (w * h) / (len(image) * len(image[0]))
        # print(ratio)
        if ratio < 1:
            # print(ratio)

            cv2.drawContours(
                new_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
            )

            # debug_image(mask,"Segments")

    return new_mask

def in_paint_optimized(
    frame: np.ndarray,
    mask: np.ndarray,
    filtered: list[tuple[tuple[int, int, int, int], str, float]] = [],
    max_height: int = 256,
    max_width: int = 256,
    mask_dilation_kernel_size: int = 9,
    inpaint_fun: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda a, b: a,
) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = frame.shape
    max_height = int(math.floor(max_height / 8) * 8)
    max_width = int(math.floor(max_width / 8) * 8)

    # only inpaint sections with masks and isolate said masks
    final = frame.copy()
    text_mask = np.zeros_like(mask)

    half_height = int(max_height / 2)
    half_width = int(max_width / 2)

    for bbox, cls, conf in filtered:
        try:
            bx1, by1, bx2, by2 = bbox
            bx1, by1, bx2, by2 = round(bx1), round(by1), round(bx2), round(by2)

            half_bx = round((bx2 - bx1) / 2)
            half_by = round((by2 - by1) / 2)
            midpoint_x, midpoint_y = round(bx1 + half_bx), round(by1 + half_by)

            x1, y1 = max(0, midpoint_x - half_width), max(0, midpoint_y - half_height)

            x2, y2 = min(w, midpoint_x + half_width), min(h, midpoint_y + half_height)

            if y2 < by2:
                y2 = by2

            if y1 > by1:
                y1 = by1

            if x2 < bx2:
                x2 = bx2

            if x1 > bx1:
                x1 = bx1

            overflow_x = (x2 - x1) % 8
            x1_adjust = 0
            if overflow_x != 0:
                if x2 > x1:
                    x2 -= overflow_x
                else:
                    x1 += overflow_x
                    x1_adjust = overflow_x

            overflow_y = (y2 - y1) % 8

            y1_adjust = 0
            if overflow_y != 0:
                if y2 > y1:
                    y2 -= overflow_y
                else:
                    y1 += overflow_y
                    y1_adjust = overflow_y

            bx1 = bx1 - (x1 + x1_adjust)
            bx2 = bx2 - (x1 + x1_adjust)
            by1 = by1 - (y1 + y1_adjust)
            by2 = by2 - (y1 + y1_adjust)

            region_mask = mask[y1:y2, x1:x2].copy()

            focus_mask = cv2.rectangle(
                np.zeros_like(region_mask),
                (bx1, by1),
                (bx2, by2),
                (255, 255, 255),
                -1,
            )

            region_mask = apply_mask(
                region_mask, np.zeros_like(region_mask), focus_mask
            )

            if has_white(region_mask):
                (
                    target_region_x1,
                    target_region_y1,
                    target_region_x2,
                    target_region_y2,
                ) = get_masked_bounds(region_mask)

                section_to_in_paint = final[y1:y2, x1:x2]

                section_to_refine = section_to_in_paint[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ]
                section_to_refine_mask = region_mask[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ]

                # Generate a mask of the actual characters/text
                refined_mask = np.zeros_like(region_mask)
                refined_mask[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ] = mask_text_for_in_painting(section_to_refine, section_to_refine_mask)

                # The text mask is used for other stuff so we set it here before we dilate for inpainting
                text_mask[y1:y2, x1:x2][
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ] = refined_mask[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ].copy()

                # Dilate the text mask for inpainting
                kernel = np.ones(
                    (mask_dilation_kernel_size, mask_dilation_kernel_size), np.uint8
                )
                refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)

                # Inpaint using the dilated text mask
                final[y1:y2, x1:x2][
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ] = inpaint_fun(final[y1:y2, x1:x2], refined_mask)[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ]
        except:
            traceback.print_exc()
            continue

    return final, text_mask


def pixels_to_pt(pixels: int):
    return pixels * 12 / 16


def pt_to_pixels(pt: int):
    return pt * (16 / 12)


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


def wrap_text(text: str, max_chars: int, hyphenator: Union[Hyphenator, None]):
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
                    pairs = hyphenator.pairs(current_word)
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


def get_fonts() -> list[tuple[str, str]]:
    fonts = []
    for file in filter(lambda a: a.endswith(".ttf"), os.listdir("./fonts")):
        fonts.append((file[0:-4], os.path.abspath(os.path.join("./fonts", file))))

    return fonts


def get_model_path(model=""):
    return os.path.join(os.path.abspath("./models"), model)


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
    hyphenator: Union[Hyphenator, None] = None,
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


def color_diff(color1: np.ndarray, color2: np.ndarray):
    # Calculate the color difference using Euclidean distance formula
    return np.sqrt(np.sum((color1 - color2) ** 2))


def draw_text_in_bubble(
    frame: np.ndarray,
    bounds: tuple[int],
    text="",
    font_file="fonts/animeace2_reg.ttf",
    color=(0, 0, 0),
    outline=1,
    outline_color: Union[tuple[int, int, int], None] = (255, 255, 255),
    hyphenator: Union[Hyphenator, None] = Hyphenator("en_US"),
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


def get_image_slices(image: np.ndarray, slice_size: tuple[int, int]):
    img_h, img_w, _ = image.shape
    slice_w, slice_h = slice_size

    parts_x = math.floor(img_w / slice_w)
    parts_y = math.floor(img_h / slice_h)

    if parts_x == 0 or parts_y == 0:  # cant slice
        return []

    parts_total = parts_x * parts_y

    slices = []

    to_slice = image.copy()

    for i in range(parts_total):
        x = i % parts_x
        y = math.floor(i / parts_total if i > 0 else 0)

        x1 = (x * slice_w) if x < parts_x else img_w - slice_w
        x2 = ((x + 1) * slice_w) if x < parts_x else img_w
        y1 = (y * slice_h) if y < parts_y else img_h - slice_h
        y2 = ((y + 1) * slice_h) if y < parts_y else img_h

        item = to_slice[y1:y2, x1:x2]

        slices.append(item)

    return slices




class COCO_TO_YOLO_TASK:
    SEGMENTATION = "seg"
    DETECTION = "detect"


def pad(num: int, amount=3):
    final = f"{num}"
    if len(final) >= amount:
        return final

    for i in range(amount - len(final)):
        final = "0" + final

    return final

def resize_and_pad(cv2_image: np.ndarray,target_size: tuple[int,int],extra_padding: int = 0,pad_color: tuple[int,int,int] = (255, 255, 255),interpolation: int = cv2.INTER_CUBIC):
    image = cv2_image.copy()
    height, width = image.shape[:2]
    max_dim = max(height, width)
    image_ratio = width / height
    target_width,target_height = target_size
    target_ratio = target_width / target_height
    should_match_height = target_ratio > image_ratio

    if should_match_height:
        
        height_factor = target_height / height
        new_width = int(height_factor * width)
        image = cv2.resize(image,(new_width,target_height),interpolation=interpolation)
    else:
        width_factor = target_width / width
        new_height = int(width_factor * height)
        image = cv2.resize(image,(target_width,new_height),interpolation=interpolation)

    height, width = image.shape[:2]

    # Calculate the amount of padding needed for each dimension
    pad_height = (target_height - height) + extra_padding
    pad_width = (target_width - width) + extra_padding

    # Determine the amount of padding on each side of the image
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    return cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )

def index_images_in_dir(index: dict, path: str):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            index_images_in_dir(index, item_path)
        else:
            index[item] = item_path
    return


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance.
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def coco_to_yolo(
    json_dir, images_dir, out_dir="new_dataset", task=COCO_TO_YOLO_TASK.SEGMENTATION
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    out_images_dir = os.path.join(out_dir, "images")
    out_labels_dir = os.path.join(out_dir, "labels")
    os.mkdir(out_dir)
    os.mkdir(out_images_dir)
    os.mkdir(out_labels_dir)

    with open(json_dir, "r") as f:
        json_file = json.load(f)
        images = json_file["images"]
        annotations = json_file["annotations"]
        coco_images_indexed = {}
        for img in tqdm(images, desc="Indexing images"):
            coco_images_indexed[img["id"]] = img

        local_images_indexed = {}

        print("Searching for local images")
        index_images_in_dir(local_images_indexed, images_dir)
        print(f"Found {len(local_images_indexed.keys())} local images")

        labels = {}
        for annotation in tqdm(annotations, desc="Converting annotations"):
            computed = []

            image = coco_images_indexed[annotation["image_id"]]
            w, h = image["width"], image["height"]
            annotation_class = str(annotation["category_id"] - 1)
            image_file_name = image["file_name"]

            fn, _ = image_file_name.split(".")
            combined_name = f"{fn}.txt|{image_file_name}"

            if combined_name not in labels.keys():
                labels[combined_name] = []

            if task == COCO_TO_YOLO_TASK.SEGMENTATION:
                if len(annotation["segmentation"]) > 1:
                    s = merge_multi_segment(annotation["segmentation"])
                    s = (
                        (np.concatenate(s, axis=0) / np.array([w, h]))
                        .reshape(-1)
                        .tolist()
                    )
                else:
                    s = [
                        j for i in annotation["segmentation"] for j in i
                    ]  # all segments concatenated
                    s = (
                        (np.array(s).reshape(-1, 2) / np.array([w, h]))
                        .reshape(-1)
                        .tolist()
                    )
                s = " ".join(
                    [annotation_class] + list(map(lambda a: str(round(a, 6)), s))
                )
                if s not in computed:
                    computed.append(s)
            elif task == COCO_TO_YOLO_TASK.DETECTION:
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(annotation["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                box = " ".join(
                    [annotation_class]
                    + list(map(lambda a: str(round(a, 6)), box.tolist()))
                )
                if box not in computed:
                    computed.append(box)

            labels[combined_name].extend(computed)

        for combined_name, computed in tqdm(labels.items(), desc="Saving Dataset"):
            out_name, image_name = combined_name.split("|")
            with open(os.path.join(out_labels_dir, out_name), "wt") as f:
                f.write("\n".join(computed))

            shutil.copy(
                local_images_indexed[image_name],
                os.path.join(out_images_dir, image_name),
            )


def roboflow_coco_to_yolo(dataset_dir):
    annotations_path = os.path.join(dataset_dir, "_annotations.coco.json")
    if not os.path.exists(annotations_path):
        for item in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, item)):
                roboflow_coco_to_yolo(os.path.join(dataset_dir, item))
        return

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    image_files = []

    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_files.append((os.path.join(dataset_dir, file), file))

    os.mkdir(images_dir)
    os.mkdir(labels_dir)

    for image_file, image_name in tqdm(image_files, desc="Moving Images"):
        shutil.move(
            image_file,
            os.path.join(images_dir, image_name),
        )

    image_files = []

    with open(annotations_path, "r") as f:
        json_file = json.load(f)
        image_files = json_file["images"]
        annotations = json_file["annotations"]
        coco_images_indexed = {}
        for img in tqdm(image_files, desc="Indexing images"):
            coco_images_indexed[img["id"]] = img

        labels = {}
        for annotation in tqdm(annotations, desc="Converting annotations"):
            computed = []

            image = coco_images_indexed[annotation["image_id"]]
            w, h = image["width"], image["height"]
            annotation_class = str(annotation["category_id"] - 1)
            image_file_name = image["file_name"]
            final_filename = image_file_name.replace(".jpg", ".txt")
            if final_filename not in labels.keys():
                labels[final_filename] = []
            if len(annotation["segmentation"]) > 1:
                s = merge_multi_segment(annotation["segmentation"])
                s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
            else:
                s = [
                    j for i in annotation["segmentation"] for j in i
                ]  # all segments concatenated
                s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
            s = " ".join([annotation_class] + list(map(lambda a: str(round(a, 6)), s)))
            if s not in computed:
                computed.append(s)

            labels[final_filename].extend(computed)

        for filename, computed in tqdm(labels.items(), desc="Saving Labels"):
            with open(os.path.join(labels_dir, filename), "wt") as J:
                J.write("\n".join(computed))

    os.remove(annotations_path)

def union(box1: tuple[int,int,int,int], box2: tuple[int,int,int,int]) -> float:
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)

def intersection(box1: tuple[int,int,int,int], box2: tuple[int,int,int,int]) -> float:
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


def overlap_area(box1: tuple[int,int,int,int], box2: tuple[int,int,int,int]):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    if box1_x2 < box2_x1 or (not (box1_y1 <= box2_y1 <= box1_y2) and not (box2_y1 <= box1_y1 <= box2_y2) and not (box1_y1 <= box2_y2 <= box1_y2) and not (box2_y1 <= box1_y2 <= box2_y2)):
        return 0
    
    return 1

def overlap_percent(box1: tuple[int,int,int,int], box2: tuple[int,int,int,int]) -> float:
    if (box2_x1 - box1_x1) < 0:
        area_overlaped = overlap_area(box2,box1)
    else:
        area_overlaped = overlap_area(box1,box2)
    
    if area_overlaped == 0:
        return 0
    

    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

def is_cuda_available():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0
