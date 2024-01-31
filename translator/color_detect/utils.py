import random
from typing import Any
import torch
import cv2
import numpy as np
from torchvision import transforms

from translator.utils import display_image, draw_text_in_bubble
from .constants import IMAGE_SIZE


class PadTensorToSquare:

    def __call__(self, image: torch.Tensor) -> Any:
        height,width = image.shape[1:]
        if height == width:
            return image
        
        max_dim = max(height,width)
        half_width = round((max_dim - width) / 2)
        half_height = round((max_dim - height) / 2)

        return torch.nn.functional.pad(image.unsqueeze(0),[half_width,half_width,half_height,half_height]).squeeze()
    
class CvToTensor:

    def __call__(self, image: np.ndarray) -> Any:
        img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb.transpose((2,0,1))).float()
    
class TensorToCv:
    def __call__(self, image: torch.Tensor) -> Any:
        img_numpy = image.cpu().numpy().transpose((1,2,0))
        img_rgb = cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR)
        return img_rgb
    
apply_transforms = transforms.Compose(
    [
        CvToTensor(),
        PadTensorToSquare(),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE),antialias=False),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ]
)

apply_transforms_inverse = transforms.Compose(
    [
        TensorToCv()
    ]
)


def random_section_from_image(
    image: np.ndarray, section_size: tuple[int, int], generator=random.Random()
):
    img_h, img_w, _ = image.shape
    section_w, section_h = section_size
    if img_h - section_h < 0 or img_w - section_w < 0:
        return None

    x1 = generator.randint(0, img_w - section_w)
    x2 = x1 + section_w
    y1 = generator.randint(0, img_h - section_h)
    y2 = y1 + section_h

    return image[y1:y2, x1:x2]


def rand_color(generator=random.Random()):
    return (
        generator.randint(0, 255),
        generator.randint(0, 255),
        generator.randint(0, 255),
    )


def get_average_color(surface: np.ndarray):
    return np.array([round(x) for x in cv2.mean(surface)][0:3])


def get_luminance(color: np.ndarray):
    """Calculates luminance of a bgr color. 1 is bright, 0 is dark"""
    b, g, r = color / 255

    return (
        (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    )  # https://en.wikipedia.org/wiki/Luminance_%28relative%29


def luminance_similarity(a: np.ndarray, b: np.ndarray):
    return 1 - abs(get_luminance(a) - get_luminance(b))


def get_outline_color(
    surface: np.ndarray, text_color: np.ndarray, min_outline_lum=0.35
):
    surface_color = get_average_color(surface)

    color_white, color_black = np.array((255, 255, 255), dtype=np.uint8), np.array(
        (0, 0, 0), dtype=np.uint8
    )

    lum_similarity = luminance_similarity(text_color, surface_color)

    # debug_image(surface,"Surface")
    if (
        lum_similarity > min_outline_lum
    ):  # we will most likely need an outline to not clash with the background
        white_similarity = luminance_similarity(text_color, color_white)

        if (
            white_similarity > 0.7
        ):  # make outline black if text is more white than black
            return tuple([x for x in color_black])
    else:
        return None

    return tuple([x for x in color_white])



# def transform_sample(cv2_image: np.ndarray, pad_img=True):
#     image = cv2_image.copy()

#     if pad_img:
#         height, width = image.shape[:2]

#         max_dim = max(height, width) + 10

#         # Calculate the amount of padding needed for each dimension
#         pad_height = max_dim - height
#         pad_width = max_dim - width

#         # Determine the amount of padding on each side of the image
#         top = pad_height // 2
#         bottom = pad_height - top
#         left = pad_width // 2
#         right = pad_width - left

#         # Set the pad color to white (255)
#         pad_color = (255, 255, 255)  # White in BGR

#         # Pad the image with the calculated padding
#         image = cv2.copyMakeBorder(
#             image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
#         )
#         # debug_image(image,"padded")

#     return apply_transforms(image)

def generate_color_detection_train_example(
    text: str = "Sample",
    background: np.ndarray = np.full((500,500, 3), 255, dtype=np.uint8),
    size: tuple[int, int] = (200, 200),
    # shift_max: tuple[int, int] = (20, 20),
    # rotation_angles: list[int] = [0, 90, -90, 180],
    outline_range: tuple[int,int] = (1,3),
    font_file="fonts/animeace2_reg.ttf",
    generator=random.Random(),
):
 

    draw_surface = random_section_from_image(
        background, size, generator
    )  # background, white or black

    if draw_surface is None:  # background was too small
        draw_surface = np.full((*size[::-1], 3), 255, dtype=np.uint8)

    draw_text_color = np.array(rand_color(generator),dtype=np.uint8)  # random color
    
    outline_color = get_outline_color(draw_surface, draw_text_color)

    frame_drawn = draw_text_in_bubble(
        draw_surface,
        ((20, 20), (draw_surface.shape[1] - 20, draw_surface.shape[0] - 20)),
        text,
        font_file=font_file,
        color=draw_text_color,
        outline_color=outline_color,
        hyphenator=None,
        outline=generator.choice(outline_range),
        # offset=[generator.randint(-1 * x, x) for x in shift_max],
        # rotation=generator.choice(rotation_angles),
    )

    has_outline =outline_color is not None
    bg_color = (np.array(outline_color,dtype=np.uint8) if has_outline else np.array([0,0,0]))
    # print("COLORS",draw_text_color,bg_color)
    # display_image(frame_drawn, "Test Frame")
    # Result is frame_drawn and text_color + has_outline 
    return frame_drawn, np.concatenate([draw_text_color,bg_color,np.array([1 if has_outline else 0])]) # np.array(draw_text_color), 
#np.concatenate([draw_text_color,bg_color,np.array([1 if has_outline else 0])])
