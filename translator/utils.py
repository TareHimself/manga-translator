import numpy as np
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import largestinteriorrectangle as lir
import os
import math
from hyphen import Hyphenator
from hyphen.textwrap2 import wrap
import shutil
import threading
from tqdm import tqdm
from .inpainting import inpaint_threadsafe
from collections import deque

en_hyphenator = Hyphenator("en_US")


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def has_white(image):
    # Set RGB values for white
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)

    # Find white pixels within the specified range
    white_pixels = cv2.inRange(image, white_lower, white_upper)

    # Check if any white pixels were found
    return cv2.countNonZero(white_pixels) > 0


def debug_image(img, name="debug"):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def ensure_gray(img):
    if len(img.shape) > 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    return img.copy()


def do_mask(a, b, mask, inv=False):
    mask = ensure_gray(mask)
    a_loc, b_loc = a.copy(), b.copy()
    mask_inv = cv2.bitwise_not(mask)

    if inv:
        temp = mask
        mask = mask_inv
        mask_inv = temp

    a_loc = cv2.bitwise_and(a_loc, a_loc, mask=mask_inv)
    b_loc = cv2.bitwise_and(b_loc, b_loc, mask=mask)
    return cv2.add(a_loc, b_loc)


def clean_text(frame, frame_mask):
    text = frame.copy()
    # cleaned = frame.copy()
    # cleaned = do_mask(
    #     cleaned,
    #     np.full(cleaned.shape, 255, dtype=cleaned.dtype),
    #     frame_mask,
    # )
    cleaned = pil_to_cv2(inpaint_threadsafe(cv2_to_pil(frame), cv2_to_pil(frame_mask)))
    text = do_mask(text, np.full(text.shape, 255, dtype=text.dtype), frame_mask, True)
    return text, cleaned


# make_bubble_mask v2
def make_bubble_mask(frame):
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


def extract_bubble(frame, frame_mask):
    text, cleaned = clean_text(frame, frame_mask)

    mask = make_bubble_mask(cleaned)

    bubble_color = np.full(cleaned.shape, 255, dtype=cleaned.dtype)

    return do_mask(cleaned, bubble_color, mask), text, mask


def generate_bubble_mask(frame, frame_text_mask, frame_cleaned):
    text = do_mask(
        frame.copy(),
        np.full(frame.shape, 255, dtype=frame.dtype),
        frame_text_mask,
        True,
    )

    return text, make_bubble_mask(frame_cleaned)


def cv2_to_pil(img) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) == 2:
        return cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    
    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_bounds_for_text(frame_mask):
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
    a1, a2, b1, b2, was_sorted, was_intersecting=False
):
    if was_sorted:
        return b1, b2, a1, a2, was_intersecting

    return a1, a2, b1, b2, was_intersecting


def fix_intersection(a1, a2, b1, b2, sorted=False, was_sorted=False):
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
    
def simplify_points(points):
    # Convert the points to a NumPy array
    points_array = np.array(points)
    
    # Find the convex hull of the points
    hull = cv2.convexHull(points_array)
    
    # Convert the convex hull back to a list of points
    simplified_points = np.array([np.array(x) for x in hull.squeeze().tolist()])
    
    return simplified_points

# def mask_charactetrs(frame: np.ndarray):

#     sample = frame.copy()

#     mser = cv2.MSER_create()
    
#     initial_size = (sample.shape[1], sample.shape[0])
#     scale_amt = max(30000 / (initial_size[0] * initial_size[1]),4)

#     sample = cv2.resize(sample, (int(sample.shape[1]*scale_amt), int(sample.shape[0]*scale_amt)))

#     final = np.zeros_like(sample,dtype=np.uint8)

#     grey_sample = cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)

#     regions,_ = mser.detectRegions(grey_sample)
    
#     sample_area = initial_size[0] * initial_size[1]

#     # debug_image(sample,"CURRENT SPECIMEN")
#     regions = list(filter(lambda a: (cv2.contourArea(a) / sample_area) < 0.6,regions))
#     # regions = [simplify_points(x) for x in regions]
#     # cv2.polylines(final, [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions], True, (0,255,0)) 
#     # cv2.drawContours(final, regions, -1, color=(255, 255, 255), thickness=cv2.FILLED)

#     cv2.polylines(final, regions,1,(255,255,255),5)

#     result = cv2.resize(final,initial_size)
#     # dilation_ammount = 4
#     # kernel = np.ones((dilation_ammount,dilation_ammount),np.uint8)
#     # img_e = cv2.dilate(cv2.resize(final,initial_size),kernel,iterations = 1)
#     return result

def mask_charactetrs(frame: np.ndarray,final_mask_dilation = 6):
    # debug_image(frame,"Data")
    image = frame.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    # Perform morphological operations to improve the text extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of the characters
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image
    mask = np.zeros_like(image)

    # Draw contours on the mask
    for contour in contours:
        # Filter out small contours and contours with a large aspect ratio
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            
    kernel = np.ones((final_mask_dilation,final_mask_dilation),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # debug_image(mask,"BETWEEN")
    # # Find contours of the characters
    # contours, heiriachy = cv2.findContours(cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY),125,255,0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Create a blank mask image
    # mask = np.zeros_like(image)

    # # Draw contours on the mask
    # for contour,info in zip(contours,heiriachy[0]):
    #     # Filter out small contours and contours with a large aspect ratio
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    #     # if info[2] == -1:
    #     #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=1)#cv2.FILLED)

    return mask


inpainting_lock = threading.Lock()

def inpaint_optimized(
    frame: np.ndarray,
    mask: np.ndarray,
    filtered: list,
    max_height=256,
    max_width=256,
):
    h, w, c = frame.shape
    max_height = int(math.floor(max_height / 8) * 8)
    max_width = int(math.floor(max_width / 8) * 8)
    
    # only inpaint sections with masks and isolate said masks
    final = frame.copy()
    text_mask = frame.copy()

    half_height = int(max_height / 2)
    half_width = int(max_width / 2)

    for bbox, cls, conf in filtered:
        bx1, by1, bx2, by2 = bbox
        bx1, by1, bx2, by2 = round(bx1), round(by1), round(bx2), round(by2)

        half_bx = round((bx2 - bx1) / 2)
        half_by = round((by2 - by1) / 2)
        midpoint_x, midpoint_y = round(bx1 + (half_bx)), round(by1 + (half_by))

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

        region_mask = do_mask(
            region_mask, np.zeros_like(region_mask), focus_mask, True
        )

        # debug_image(region_mask,"Current Region Mask")

        # debug_image(final[y1:y2, x1:x2],"Current Region Image")

        if has_white(region_mask):
            
            section = final[y1:y2, x1:x2]
            # debug_image(section,"Target")
            # debug_image(region_mask,"Initial Mask")
            refined_mask = mask_charactetrs(do_mask(section,np.ones_like(region_mask) * 255,region_mask,True))
            # debug_image(refined_mask,"Refined mask")
            final[y1:y2, x1:x2] = pil_to_cv2(
                inpaint_threadsafe(cv2_to_pil(final[y1:y2, x1:x2]), cv2_to_pil(refined_mask))
            )

            text_mask[y1:y2, x1:x2] = refined_mask

            # final[y1:y2, x1:x2] = do_mask(final[y1:y2, x1:x2],np.ones_like(final[y1:y2, x1:x2]) * 255,refined_mask,False)

            # final[y1:y2, x1:x2] = pil_to_cv2(
            #     inpaint(cv2_to_pil(final[y1:y2, x1:x2]), cv2_to_pil(region_mask))
            # )

    return final,text_mask


def pixels_to_pt(pixels):
    return pixels * 12 / 16


def pt_to_pixels(pt):
    return pt * (16 / 12)


def try_merge_hypnenated(text: list[str], max_chars: int):
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


def wrap_text(text: str, max_chars: int):
    total = deque(text.split(" "))
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
                else:
                    pairs = en_hyphenator.pairs(current_word)
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

    return try_merge_hypnenated(lines, max_chars)


# [print(x, en_hyphenator.pairs(x)) for x in text.split(" ")]
# return wrap(text, max_chars, use_hyphenator=en_hyphenator)

def get_fonts():
    fonts = []
    idx = 0
    for file in filter(lambda a: a.endswith('.ttf'),os.listdir("./fonts")):
        fonts.append({
            "id": idx,
            "name": file[0:-4]
        })

        idx += 1
    
    return fonts

def get_font_path_at_index(idx: int):
    return os.path.join("./fonts",list(filter(lambda a: a.endswith('.ttf'),os.listdir("./fonts")))[idx])

def get_average_font_size(font, text="some text here"):
    x, y, w, h = font.getbbox(text)
    widths = list(map(lambda a: font.getbbox(a)[2], list(text)))
    widths.sort(reverse=True)
    return widths[1] if len(widths) > 1 else widths[0], h


def get_best_font_size(
    text,
    wh,
    font_file,
    space_between_lines=1,
    start_size=18,
    step=1,
    min_chars_per_line=6,
    initial_iterations=0,
):
    current_font_size = start_size
    current_font = None
    max_width, max_height = wh

    iterations = initial_iterations
    while True:
        iterations += 1

        if current_font_size < 0:
            return None, None, None, iterations

        # if current_font_size < 6:
        #     print(
        #         "REDUCING MIN CHARACTERS FROM",
        #         min_chars_per_line,
        #         "TO",
        #         min_chars_per_line - 1,
        #     )
        #     return get_best_font_size(
        #         text,
        #         wh,
        #         font_file,
        #         space_between_lines,
        #         start_size,
        #         step,
        #         min_chars_per_line - 1,
        #         iterations,
        #     )
        current_font = ImageFont.truetype(font_file, current_font_size)
        cur_f_width, cur_f_height = get_average_font_size(current_font, text)
        chars_per_line = math.floor(max_width / cur_f_width)
        # print(
        #     "DEBUG",
        #     chars_per_line,
        #     max_width,
        #     cur_f_width,
        #     current_font_size,
        #     min_chars_per_line,
        # )
        if chars_per_line < min_chars_per_line:
            current_font_size -= step
            continue

        # print(chars_per_line)
        lines = wrap_text(text, chars_per_line)
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
    frame,
    bounds,
    text="",
    font_file="fonts/animeace2_reg.ttf"
):
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
    frame_to_pil = cv2_to_pil(frame)
    draw = ImageDraw.Draw(frame_to_pil)

    font = ImageFont.truetype("fonts/BlambotClassicBB.ttf", font_size)
    draw_x = pt1[0]
    draw_y = pt1[1]

    wrapped = wrap_text(text, chars_per_line)
    
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
            fill=(0, 0, 0, 255),
            font=font,
        )

    return pil_to_cv2(frame_to_pil)


class COCO_TO_YOLO_TASK:
    SEGMENTATION = "seg"
    DETECTION = "detect"


def pad(num: int, ammount=3):
    final = f"{num}"
    if len(final) >= ammount:
        return final

    for i in range(ammount - len(final)):
        final = "0" + final

    return final


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
            with open(os.path.join(labels_dir, filename), "wt") as f:
                f.write("\n".join(computed))

    os.remove(annotations_path)
