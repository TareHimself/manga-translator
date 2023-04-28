import numpy as np
import easyocr
import cv2
import json
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import largestinteriorrectangle as lir
import textwrap
import math
import hyphen

en_hyphenator = hyphen.Hyphenator("en_US")

en_hyphenator.wrap


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def debug_image(img, name="debug"):
    cv2.imshow(name, img)
    cv2.waitKey(0)


# def clean_text(frame):
#     total_tl = [9e12, 9e12]
#     total_br = [0, 0]

#     results = ocr.readtext(frame)

#     if len(results) == 0:
#         return frame, frame

#     cleaned = frame.copy()

#     for result in results:
#         tl, _, br, __ = result[0]

#         x1, y1 = int(tl[0]), int(tl[1])
#         x2, y2 = int(br[0]), int(br[1])

#         total_tl[0] = x1 if x1 < total_tl[0] else total_tl[0]
#         total_tl[1] = y1 if y1 < total_tl[1] else total_tl[1]
#         total_br[0] = x2 if x2 > total_br[0] else total_br[0]
#         total_br[1] = y2 if y2 > total_br[1] else total_br[1]

#         cv2.rectangle(cleaned, (x1, y1), (x2, y2), (255, 255, 255), -1)

#     x1, y1 = int(total_tl[0]), int(total_tl[1])
#     x2, y2 = int(total_br[0]), int(total_br[1])
#     text = frame.copy()
#     text = text[y1:y2, x1:x2]

#     return text, cleaned


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
    cleaned = frame.copy()
    text = frame.copy()
    cleaned = do_mask(
        cleaned,
        np.full(cleaned.shape, 255, dtype=cleaned.dtype),
        frame_mask,
    )
    text = do_mask(text, np.full(text.shape, 255, dtype=text.dtype), frame_mask, True)
    return text, cleaned


def make_bubble_mask(frame):
    item = frame.copy()
    # Convert to grayscale and apply adaptive thresholding
    gray_frame = cv2.cvtColor(item, cv2.COLOR_RGB2GRAY)

    # gray_frame = cv2.GaussianBlur(gray_frame, (19, 19), 0)

    # cv2.imshow("blurred", adjust_contrast_brightness(gray_frame, 20))
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    # # see the results
    # cv2.imshow("All Contours", image_copy)
    # cv2.waitKey(0)
    # Create an empty mask for the bubble
    mask = np.zeros_like(gray_frame)

    # Fill in the mask with the largest contour (assuming it's the bubble)

    # cv2.imshow(
    #     "all contours",
    #     cv2.drawContours(
    #         image=mask.copy(),
    #         contours=contours,
    #         contourIdx=-1,
    #         color=(255, 255, 255),
    #         thickness=2,
    #         lineType=cv2.LINE_AA,
    #     ),
    # )

    # cv2.waitKey(0)

    image_area = len(mask) * len(mask[0])
    best = None
    latest = 0
    for cnt in contours:
        coverage = (
            cv2.fillPoly(mask.copy(), pts=[cnt], color=(255, 255, 255)) == 255
        ).sum() / image_area
        if coverage > latest:
            best = cnt
            latest = coverage

    cv2.fillPoly(mask, pts=[best], color=(255, 255, 255))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # draw contours on the original image

    # masked_bubble = cv2.bitwise_and(bubble, bubble, mask=mask)

    return mask


# make_bubble_mask v2
def mask_frame(frame):
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

    # # Find contours in the image
    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # # Create a black image with the same size as the original

    # stage_1 = cv2.drawContours(
    #     np.zeros_like(image), contours, -1, (255, 255, 255), thickness=1
    # )

    # # ret, thresh = cv2.threshold(stage_1, 200, 255, cv2.THRESH_BINARY)

    # stage_1 = cv2.cvtColor(stage_1, cv2.COLOR_BGR2GRAY)

    # cv2.imshow(
    #     "state 1 contour",
    #     stage_1,
    # )
    # cv2.waitKey(0)

    # # Find the largest contour, assuming it's the circle
    # max_contour = max(contours, key=cv2.contourArea)

    # # Draw the largest contour (circle) filled with white color

    # # Find contours in the image
    # contours, _ = cv2.findContours(stage_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # mask = np.zeros_like(image)

    # for cont in contours:
    #     cv2.imshow(
    #         "PARTS", cv2.fillPoly(mask.copy(), pts=[cont], color=(255, 255, 255))
    #     )
    #     cv2.waitKey(0)

    # cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return mask


def extract_bubble(frame, frame_mask):
    text, cleaned = clean_text(frame, frame_mask)

    mask = mask_frame(cleaned)

    bubble_color = np.full(cleaned.shape, 255, dtype=cleaned.dtype)

    return do_mask(cleaned, bubble_color, mask), text, mask


def cv2_to_pil(img) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img) -> np.ndarray:
    return np.array(img)[:, :, ::-1]


def get_text_area(frame_mask):
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

    return rect, lir.pt1(rect), lir.pt2(rect)


def pixels_to_pt(pixels):
    return pixels * 12 / 16


def pt_to_pixels(pt):
    return pt * (16 / 12)


def wrap_text(text, max_chars):
    result = []
    text = text.split(" ")
    if len(max(text, key=len)) > max_chars:
        return False, []

    current_line = ""
    for word in text:
        if len(current_line + word) > max_chars:
            result.append(current_line)
            current_line = word
        else:
            current_line += " " + word
            current_line = current_line.strip()
    if len(current_line):
        result.append(current_line)
    return True, result


def get_average_font_size(font, text="some text here"):
    x, y, w, h = font.getbbox(text)
    widths = list(map(lambda a: font.getbbox(a)[2], list(text)))
    widths.sort(reverse=True)
    return widths[1] if len(widths) > 1 else widths[0], h


def get_best_font_size(
    text, wh, font_file, space_between_lines=1, start_size=18, step=1
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
        was_successful, lines = wrap_text(text, chars_per_line)

        if not was_successful:
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
    frame_mask,
    text="Lorem ipsum dolor sit amet,  elit.",
):
    rect, pt1, pt2 = get_text_area(frame_mask)

    max_height = rect[3]
    max_width = rect[2]

    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 1)

    space_between_lines = 0
    font_size, chars_per_line, line_height, iters = get_best_font_size(
        text,
        (max_width, max_height),
        "fonts/BlambotClassicBB.ttf",
        space_between_lines,
        90,
        1,
    )

    # print(text, font_size, chars_per_line, line_height, iters)
    if not font_size:
        return frame
    frame_to_pil = cv2_to_pil(frame)
    draw = ImageDraw.Draw(frame_to_pil)

    font = ImageFont.truetype("fonts/BlambotClassicBB.ttf", font_size)
    draw_x = pt1[0]
    draw_y = pt1[1]

    successful, wrapped = wrap_text(text, chars_per_line)
    for line_no in range(len(wrapped)):
        line = wrapped[line_no]
        x, y, w, h = font.getbbox(line)
        draw.text(
            (
                draw_x + abs(((max_width - w) / 2)),
                draw_y
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


def coco_to_yolo(json_dir, images_dir, task=COCO_TO_YOLO_TASK.DETECTION):
    with open(json_dir, "r") as f:
        json_file = json.load(f)
        images = json_file["images"]
        annotations = json_file["annotations"]
        categories = json_file["categories"]
        images_index = {}
        for img in images:
            images_index[img["id"]] = img

        categories_index = {}
        for cat in categories:
            categories_index[cat["id"]] = cat
