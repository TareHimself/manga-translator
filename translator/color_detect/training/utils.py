import random
from typing import Any
import torch
import cv2
import numpy as np
from torchvision import transforms
from translator.utils import draw_text_in_bubble
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
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE),antialias=False,),
        transforms.Normalize([0,0,0], [255,255,255]),
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
    # if generator.choice([True,False]):
    #     return (0,0,0)
    # else:
    #     return (255,255,255)
    # return (
    #         generator.randint(0, 255),
    #         generator.randint(0, 255),
    #         generator.randint(0, 255),
    #     )
    idx = generator.randint(0,2)

    if idx == 0:
        return (0,0,0)
    elif idx == 1:
        return (255,255,255)
    else:
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
    surface: np.ndarray, text_color: np.ndarray, min_outline_lum=0.35,force_outline = False
):
    surface_color = get_average_color(surface)

    color_white, color_black = np.array((255, 255, 255), dtype=np.uint8), np.array(
        (0, 0, 0), dtype=np.uint8
    )

    # we could use LAB here instead
    if force_outline:
        white_similarity = luminance_similarity(text_color, color_white)
        black_similarity = luminance_similarity(text_color, color_black)

        if white_similarity >= black_similarity:
            return tuple([x for x in color_black])
        else:
            return tuple([x for x in color_white])
    lum_similarity = luminance_similarity(text_color, surface_color)

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


def generate_color_detection_train_example(
    text: str = "Sample",
    background: np.ndarray = np.full((500,500, 3), 255, dtype=np.uint8),
    size: tuple[int, int] = (200, 200),
    # shift_max: tuple[int, int] = (20, 20),
    # rotation_angles: list[int] = [0, 90, -90, 180],
    outline_range: tuple[int,int] = (1,2),
    font_file="fonts/animeace2_reg.ttf",
    generator=random.Random(),
    force_outline = False
):
 

    draw_surface = random_section_from_image(
        background, size, generator
    )  # background, white or black

    if draw_surface is None:  # background was too small
        draw_surface = np.full((*size[::-1], 3), 255, dtype=np.uint8)

    draw_text_color_tuple = rand_color(generator)
    draw_text_color = np.array(draw_text_color_tuple,dtype=np.uint8)  # random color
    
    outline_color = get_outline_color(draw_surface, draw_text_color,force_outline=force_outline)

    # outline_color = rand_color(generator) if force_outline else None
    # #print(outline_color,draw_text_color)
    # while outline_color is not None and outline_color == draw_text_color_tuple:
    #     outline_color = rand_color(generator)

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

    has_outline = outline_color is not None
    bg_color = (np.array(outline_color,dtype=np.uint8) if has_outline else draw_text_color)
    # print("COLORS",draw_text_color,bg_color)
    # display_image(frame_drawn, "Test Frame")
    # Result is frame_drawn and text_color + has_outline 
    return frame_drawn, np.concatenate([draw_text_color,bg_color,np.array([1 if has_outline else 0])]) # np.array(draw_text_color), 

def rgb_to_lab_np(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0

    # Gamma correction
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # Linear RGB to XYZ
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    xyz = rgb @ M.T
    xyz /= np.array([0.95047, 1.00000, 1.08883])  # Normalize

    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1)

def lab_to_rgb_np(lab):
    lab = np.asarray(lab, dtype=np.float32)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    fx3 = fx ** 3
    fy3 = fy ** 3
    fz3 = fz ** 3

    xr = np.where(fx3 > epsilon, fx3, (116 * fx - 16) / kappa)
    yr = np.where(fy3 > epsilon, fy3, (116 * fy - 16) / kappa)
    zr = np.where(fz3 > epsilon, fz3, (116 * fz - 16) / kappa)

    xyz = np.stack([xr, yr, zr], axis=-1)
    xyz *= np.array([0.95047, 1.00000, 1.08883])

    # XYZ to linear RGB
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb = xyz @ M.T

    # Gamma compression
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)

def normalize_lab_np(lab: np.ndarray) -> np.ndarray:
    """
    Normalize Lab values from standard scale to approx. [-1, 1] range.
    
    Args:
        lab (np.ndarray): shape (N, 3), Lab values
                          L ∈ [0, 100], a/b ∈ [-128, 127]

    Returns:
        np.ndarray: shape (N, 3), normalized Lab
                    L ∈ [-1, 1], a/b ∈ [-1, 1]
    """
    assert lab.shape[1] == 3, "Input must be of shape (N, 3)"
    L = (lab[:, 0] - 50.0) / 50.0
    a = lab[:, 1] / 128.0
    b = lab[:, 2] / 128.0
    return np.stack((L, a, b), axis=1)

def denormalize_lab_np(norm_lab: np.ndarray) -> np.ndarray:
    """
    Convert normalized Lab values back to standard Lab scale.
    
    Args:
        norm_lab (np.ndarray): shape (N, 3), normalized Lab values

    Returns:
        np.ndarray: shape (N, 3), denormalized Lab
                    L ∈ [0, 100], a/b ∈ [-128, 127]
    """
    assert norm_lab.shape[1] == 3, "Input must be of shape (N, 3)"
    L = norm_lab[:, 0] * 50.0 + 50.0
    a = norm_lab[:, 1] * 128.0
    b = norm_lab[:, 2] * 128.0
    return np.stack((L, a, b), axis=1)

def rgb_to_hsv(rgb):
    rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero_delta = delta > 1e-6  # Mask for non-zero delta

    # Hue calculation (only where delta > 0)
    mask = nonzero_delta
    r_eq_max = (maxc == r) & mask
    g_eq_max = (maxc == g) & mask
    b_eq_max = (maxc == b) & mask

    h[r_eq_max] = ((g[r_eq_max] - b[r_eq_max]) / delta[r_eq_max]) % 6
    h[g_eq_max] = ((b[g_eq_max] - r[g_eq_max]) / delta[g_eq_max]) + 2
    h[b_eq_max] = ((r[b_eq_max] - g[b_eq_max]) / delta[b_eq_max]) + 4
    h = (h / 6.0) % 1.0  # Normalize hue to [0, 1]

    # Saturation calculation (where maxc > 0)
    s[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]

    return np.stack([h, s, v], axis=1)
    # rgb = rgb.astype(np.float32) / 255.0  # Normalize RGB to [0, 1]
    # r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # maxc = np.maximum(np.maximum(r, g), b)
    # minc = np.minimum(np.minimum(r, g), b)
    # delta = maxc - minc
    # delta = np.maximum(delta,np.full_like(delta,0.00000001))
    # h = np.zeros_like(maxc)
    # s = np.zeros_like(maxc)
    # v = maxc

    # mask = delta != 0
    # rc = (((g - b) / delta) % 6)[mask]
    # gc = (((b - r) / delta) + 2)[mask]
    # bc = (((r - g) / delta) + 4)[mask]

    # h[mask & (maxc == r)] = rc
    # h[mask & (maxc == g)] = gc
    # h[mask & (maxc == b)] = bc
    # h = (h / 6.0) % 1.0  # Normalize hue to [0, 1]

    # s[mask] = delta[mask] / np.maximum(maxc[mask],np.full_like(maxc[mask],0.00000001))

    # return np.stack([h, s, v], axis=1)

def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = h * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    i = i % 6
    for idx in range(6):
        mask = i == idx
        if idx == 0:
            r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
        elif idx == 1:
            r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
        elif idx == 2:
            r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
        elif idx == 3:
            r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
        elif idx == 4:
            r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
        elif idx == 5:
            r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    rgb = np.stack([r, g, b], axis=1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)

def normalize_hsv(hsv):
    """
    Normalize HSV values:
    - H in degrees [0, 360) → [0, 1]
    - S, V in percent [0, 100] → [0, 1]
    
    Input shape: (batch, 3)
    Output shape: (batch, 3)
    """
    h = hsv[:, 0] / 360.0
    s = hsv[:, 1] / 100.0
    v = hsv[:, 2] / 100.0
    return np.stack([h, s, v], axis=1)

def denormalize_hsv(normalized_hsv):
    """
    Denormalize HSV values from [0, 1] back to:
    - H in [0, 360)
    - S, V in [0, 100]

    Input shape: (batch, 3)
    Output shape: (batch, 3)
    """
    h = normalized_hsv[:, 0] * 360.0
    s = normalized_hsv[:, 1] * 100.0
    v = normalized_hsv[:, 2] * 100.0
    return np.stack([h, s, v], axis=1)
def format_color_detect_output(result: torch.Tensor) -> list[tuple[np.ndarray,np.ndarray,bool]]:
    all_results = []
    
    # for colors,bg in zip(result[0].cpu().numpy(),result[1].cpu().numpy()):
    #     colors = np.split(colors,[3])
    #     colors = np.stack(colors)
    #     colors = denormalize_lab_np(colors)
    #     colors = lab_to_rgb_np(colors)
    #     has_outline = True if bg > 0.5 else False
    #     all_results.append((colors,has_outline))

    for colors in result.cpu().numpy():
        colors = np.array([colors])
        colors = denormalize_lab_np(colors)
        colors = lab_to_rgb_np(colors)
        all_results.append((colors[0],np.array([255,255,255],dtype=np.uint8),False))
    
    return all_results
#np.concatenate([draw_text_color,bg_color,np.array([1 if has_outline else 0])])
