import colorsys
from contextlib import nullcontext
import cv2
import langcodes
import numba as nb
import torch
import numpy as np
import pyphen
from more_itertools import split_when
from typing import Awaitable, Callable, Optional, ParamSpec, TypeVar, overload
import largestinteriorrectangle as lir
from PIL import Image, ImageFont
from comic_localizer.core.typing import Vector4i, Vector3u8
import functools
import time
from math import floor, ceil

P = ParamSpec("P")
R = TypeVar("R")

_perf_enabled = False


class PerfContext:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        if _perf_enabled:
            self.start = time.perf_counter()
        else:
            self.start = None

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if _perf_enabled and self.start is not None:
            elapsed = time.perf_counter() - self.start
            print(f"[{elapsed:.6f}s]: {self.label}")

    async def __aenter__(self):
        if _perf_enabled:
            self.start = time.perf_counter()
        else:
            self.start = None

        return self

    async def __aexit__(self, exc_type, exc, tb):
        if _perf_enabled and self.start is not None:
            elapsed = time.perf_counter() - self.start
            print(f"[{elapsed:.6f}s]: {self.label}")


@overload
def perf_async(
    maybe_original_function: Callable[P, Awaitable[R]],
    name_override: Optional[str] = None,
) -> Callable[P, Awaitable[R]]: ...


@overload
def perf_async(
    maybe_original_function: None = None,
    name_override: Optional[str] = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def perf_async(
    maybe_original_function: Optional[Callable[P, Awaitable[R]]] = None,
    name_override: Optional[str] = None,
) -> (
    Callable[P, Awaitable[R]]
    | Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]
):

    def _decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        func_name = name_override or function.__name__

        @functools.wraps(function)
        async def do_perf(*args: P.args, **kwargs: P.kwargs) -> R:
            if _perf_enabled:
                label = ""
                if "." in function.__qualname__:
                    label = f"{args[0].__class__.__name__}.{func_name}"
                else:
                    label = func_name

                label = " > ".join(label.split("."))

                async with PerfContext(label):
                    return await function(*args, **kwargs)
            else:
                return await function(*args, **kwargs)

        return do_perf

    if maybe_original_function is not None:
        return _decorate(maybe_original_function)

    return _decorate


@overload
def perf(
    maybe_original_function: Callable[P, R],
    name_override: Optional[str] = None,
) -> Callable[P, R]: ...


@overload
def perf(
    maybe_original_function: None = None,
    name_override: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def perf(
    maybe_original_function: Optional[Callable[P, R]] = None,
    name_override: Optional[str] = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:

    def _decorate(function: Callable[P, R]) -> Callable[P, R]:
        func_name = name_override or function.__name__

        @functools.wraps(function)
        def do_perf(*args: P.args, **kwargs: P.kwargs) -> R:
            if _perf_enabled:
                label = ""
                if "." in function.__qualname__:
                    label = f"{args[0].__class__.__name__}.{func_name}"
                else:
                    label = func_name

                label = " > ".join(label.split("."))

                with PerfContext(label):
                    return function(*args, **kwargs)
            else:
                return function(*args, **kwargs)

        return do_perf

    if maybe_original_function is not None:
        return _decorate(maybe_original_function)

    return _decorate


def disable_perf():
    global _perf_enabled
    _perf_enabled = False


def enable_perf():
    global _perf_enabled
    _perf_enabled = True


class WrappedLine:
    def __init__(self, words: list[str], offset: float, height: float = 0):
        self.words = words
        self.offset = offset
        self.height = height

    def add_word(self, word: str, word_height: float):
        self.words.append(word)
        self.height = max(self.height, word_height)


class WrapResult:
    def __init__(self, lines: list[WrappedLine], bounds: tuple[int | float, int | float]):
        self.lines = lines
        self.bounds = bounds


def bbox_to_rect(bbox: tuple[float, float, float, float]):
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


class LayoutCache:
    def __init__(self, font: ImageFont.FreeTypeFont, outline_size: int = 0):
        self.font = font
        self.cache = {}
        self.outline_size = outline_size

    def get(self, text: str) -> tuple[float, float, float, float]:
        if text in self.cache:
            return self.cache[text]
        rect = bbox_to_rect(self.font.getbbox(text))
        outline_width = sum(self.outline_size for char in text if char != " ")
        self.cache[text] = (
            rect[0],
            rect[1],
            rect[2] + outline_width,
            rect[3] + (self.outline_size * 2),
        )

        return self.cache[text]


class HyphenationCache:
    def __init__(
        self, hyphenator: pyphen.Pyphen, wrap: float, layout_cache: LayoutCache
    ):
        self.hyphenator = hyphenator
        self.cache = {}
        self.wrap = wrap
        self.layout_cache = layout_cache

    def add_dashes_to_hypenations(self, hyphenation: tuple[str, str]):
        return [
            (f"{hyphenation[0]}-", self.layout_cache.get(f"{hyphenation[0]}-")),
            (hyphenation[1], self.layout_cache.get(hyphenation[1])),
        ]

    def filter_out_impossible(self, hyphenations: list[list[tuple[str, tuple[float, float, float, float]]]]):
        return filter(
            lambda hyp: max(hyp, key=lambda item: item[1][2])[1][2] <= self.wrap,
            hyphenations,
        )

    def get(
        self, text: str
    ) -> list[list[tuple[str, tuple[float, float, float, float]]]]:
        if text in self.cache:
            return self.cache[text]

        self.cache[text] = list(
            self.filter_out_impossible(
                [
                    [(text, self.layout_cache.get(text))],
                    *map(self.add_dashes_to_hypenations, self.hyphenator(text)),
                ]
            )
        )

        return self.cache[text]


def has_white(image: np.ndarray):
    # Set RGB values for white
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)

    # Find white pixels within the specified range
    white_pixels = cv2.inRange(image, white_lower, white_upper)

    # Check if any white pixels were found
    return cv2.countNonZero(white_pixels) > 0


def natural_sort_key(value: str) -> list:
    groups = split_when(value, lambda a, b: a.isdigit() != b.isdigit())
    return [
        int("".join(group)) if group[0].isdigit() else "".join(group).lower()
        for group in groups
    ]


def wrap_text_pure(
    text: str,
    font: ImageFont.FreeTypeFont,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
    outline_size: int = 0,
) -> Optional[WrapResult]:
    layout_cache = LayoutCache(font=font, outline_size=outline_size)
    _, _, space_width, _ = layout_cache.get(" ")
    text_list = text.split()
    text_bounds = [(a, layout_cache.get(a)) for a in text_list]
    x_offset = 0
    # Text too big to fit on a line
    if any(a[1][2] > wrap_width for a in text_bounds):
        return None

    x_offset = 0
    line_idx = 0
    lines = [WrappedLine([], 0)]
    x_bounds = 0
    for word, bbox in text_bounds:
        x_end = x_offset + bbox[2]

        if x_end > wrap_width:
            last_line = lines[-1]
            lines.append(
                WrappedLine([], last_line.offset + last_line.height + line_spacing)
            )
            line_idx += 1

            x_bounds = max(x_bounds, x_offset)

            x_offset = 0
            x_end = bbox[2]

        lines[line_idx].add_word(word, bbox[3])
        x_offset = min(x_end + space_width, wrap_width)
        x_bounds = max(x_bounds, x_offset)

    last_line = lines[-1]
    return WrapResult(lines, (int(x_bounds), int(last_line.offset + last_line.height)))


def wrap_text_with_hyphenator(
    text: str,
    font: ImageFont.FreeTypeFont,
    hyphenator: pyphen.Pyphen,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
    outline_size: int = 0,
) -> Optional[WrapResult]:
    layout_cache = LayoutCache(font=font, outline_size=outline_size)
    hyphenation_cache = HyphenationCache(
        hyphenator=hyphenator, layout_cache=layout_cache, wrap=wrap_width
    )
    _, _, space_width, _ = layout_cache.get(" ")
    text_list = text.split()
    all_word_versions = list(map(hyphenation_cache.get, text_list))

    # No versions means the word cant fit at this font size
    if any(len(a) == 0 for a in all_word_versions):
        return None

    # we know one version of the word will fit, we just need to find the version
    def fit_best_version(
        lines: list[WrappedLine],
        versions: list[list[tuple[str, tuple[float, float, float, float]]]],
        x_offset: float,
        x_bounds: float,
    ):
        nonlocal wrap_width
        nonlocal space_width
        nonlocal line_spacing
        line_idx = len(lines) - 1

        selected_version = versions[0]
        version_part_index = 0
        # if we are at a new line we can skip this section
        if x_offset != 0:
            for version in versions:
                word_partial, bbox = version[version_part_index]
                x_end = x_offset + bbox[2]

                if x_end <= wrap_width:
                    lines[line_idx].add_word(word_partial, bbox[3])
                    x_bounds = max(x_bounds, x_end)
                    version_part_index += 1
                    selected_version = version
                    x_offset = x_end + space_width
                    break

        if version_part_index < len(selected_version):
            # now we start fitting a new line

            if len(lines[line_idx].words) > 0:
                last_line = lines[-1]
                lines.append(
                    WrappedLine([], last_line.offset + last_line.height + line_spacing)
                )
                line_idx += 1

            x_offset = 0

            for version_part in selected_version[version_part_index:]:
                word_partial, bbox = version_part

                x_end = x_offset + bbox[2]

                if x_end > wrap_width:
                    last_line = lines[-1]
                    lines.append(
                        WrappedLine(
                            [], last_line.offset + last_line.height + line_spacing
                        )
                    )
                    line_idx += 1

                    x_bounds = max(x_bounds, x_offset)

                    x_offset = 0
                    x_end = bbox[2]

                lines[line_idx].add_word(word_partial, bbox[3])
                x_offset = min(x_end + space_width, wrap_width)

                x_bounds = max(x_bounds, x_offset)

        return x_bounds, x_offset

    x_offset = 0
    x_bounds = 0
    lines = [WrappedLine([], 0)]

    for versions in all_word_versions:
        new_bounds, new_offset = fit_best_version(lines, versions, x_offset, x_bounds)
        x_bounds = new_bounds
        x_offset = new_offset

    last_line = lines[-1]
    return WrapResult(lines, (int(x_bounds), int(last_line.offset + last_line.height)))


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    hyphenator: Optional[pyphen.Pyphen] = None,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
    outline_size: int = 0,
) -> Optional[WrapResult]:
    return (
        wrap_text_pure(text, font, wrap_width, line_spacing, outline_size)
        if hyphenator is None
        else wrap_text_with_hyphenator(
            text, font, hyphenator, wrap_width, line_spacing, outline_size
        )
    )


class FontFitResult:
    def __init__(self, font_size: float, wrap: WrapResult):
        self.font_size = font_size
        self.wrap = wrap


def find_next_test(min_size, max_size):
    return min_size + ((max_size - min_size) // 2)


@functools.lru_cache()
def load_font(font_file: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_file, size=size)


# @perf
def find_best_font_size(
    text: str,
    font_file: str,
    size: tuple[int, int],
    font_size=30,
    min_font_size=10,
    max_font_size=20,
    tolerance=1,
    line_spacing: float = 2,
    hyphenator: Optional[pyphen.Pyphen] = None,
    outline_size: int = 0,
) -> Optional[FontFitResult]:
    current_size = min(font_size, max_font_size)
    current_max = max_font_size
    current_min = min_font_size
    best = None
    while current_min <= current_max:
        font = load_font(font_file, current_size)
        wrap_result = wrap_text(
            text=text,
            font=font,
            hyphenator=hyphenator,
            wrap_width=size[0],
            line_spacing=line_spacing,
            outline_size=outline_size,
        )
        if wrap_result is not None and wrap_result.bounds[1] <= size[1]:
            best = FontFitResult(current_size, wrap_result)
            current_min = current_size + 1
            next_font_size = find_next_test(current_min, current_max)

            if abs(best.font_size - next_font_size) < tolerance:
                break

            current_size = next_font_size

        else:
            current_max = current_size - 1
            next_font_size = find_next_test(current_min, current_max)
            best_font_size = current_size if best is None else best.font_size

            if abs(best_font_size - next_font_size) < tolerance:
                break

            current_size = next_font_size

    return best


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    return arr


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


@nb.njit("int64[:](boolean[:,::1])", cache=True)
def maximal_rectangle(grid: np.ndarray) -> np.ndarray:
    """
    Largest axis-aligned all-true rectangle in a boolean grid, via the
    classic histogram + monotonic stack algorithm. Exact, O(rows * cols).
    Returns [x, y, width, height], same format as largestinteriorrectangle's
    lir() output. Unlike lir.lir(polygon) (see the commented out code in
    compute_draw_bbox below) this operates directly on the grid, so it can't
    miss interior obstacles that don't touch the outer boundary.
    """
    h, w = grid.shape
    heights = np.zeros(w, dtype=np.int64)
    best_area = 0
    best_x = 0
    best_y = 0
    best_w = 0
    best_h = 0

    stack_idx = np.zeros(w + 1, dtype=np.int64)
    stack_h = np.zeros(w + 1, dtype=np.int64)

    for y in range(h):
        for x in range(w):
            if grid[y, x]:
                heights[x] += 1
            else:
                heights[x] = 0

        sp = 0
        for i in range(w + 1):
            cur_h = heights[i] if i < w else 0
            start = i
            while sp > 0 and stack_h[sp - 1] > cur_h:
                sp -= 1
                idx = stack_idx[sp]
                height = stack_h[sp]
                width = i - idx
                area = height * width
                if area > best_area:
                    best_area = area
                    best_w = width
                    best_h = height
                    best_x = idx
                    best_y = y - height + 1
                start = idx
            stack_idx[sp] = start
            stack_h[sp] = cur_h
            sp += 1

    return np.array([best_x, best_y, best_w, best_h], dtype=np.int64)


def compute_draw_bbox(section: np.ndarray) -> Vector4i:
    grey = ensure_gray(section)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    height, width = grey.shape[:2]

    min_dim = min(height, width)

    scale_to = 256
    scale = 1
    # Scaling for consistent kernel ops
    if min_dim != scale_to:
        scale = scale_to / float(min_dim)

    if scale != 1:
        new_width = round(width * scale)
        new_height = round(height * scale)
        grey = cv2.resize(
            grey,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA,
        )
        height = new_height
        width = new_width

    ret, thresh = cv2.threshold(grey, 200, 255, 0)

    mostlyWhite = np.mean(thresh > 127) > 0.5

    # if an image has a lot of white the actual bubble is probably black so we would want to invert it because morphology will cleanup white areas
    if mostlyWhite:
        thresh = 255 - thresh

    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed = cv2.dilate(morphed, kernel)

    padding = 4
    padded = np.zeros(
        (height + (padding * 2), width + (padding * 2)), dtype=morphed.dtype
    )

    start = padding
    # pad and invert since bubble is probably black and cv2.findContours needs it white
    padded[start : start + height, start : start + width] = 255 - morphed

    # Previous approach: extract the outer contour and hand it straight to
    # largestinteriorrectangle.
    # with PerfContext("compute_draw_bbox > find_contours"):
    #     contours, hierarchy = cv2.findContours(
    #         padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #
    # if len(contours) == 0:
    #     return np.array([0, 0, width, height], dtype=np.int32)
    #
    # largest_contour = max(contours, key=cv2.contourArea)[:, 0, :]
    #
    # if len(largest_contour) < 2:
    #     return np.array([0, 0, width, height], dtype=np.int32)
    #
    # polygon = np.array([largest_contour], dtype=np.int32)
    #
    # with PerfContext(f"compute_draw_bbox > lir ({len(largest_contour)} pts)"):
    #     rect = lir.lir(polygon)

    # bubbles are treated as potentially transparent, so we only care about
    # the outer silhouette here -- whatever is inside (background showing
    # through, or minor cleaning residue) gets painted over by the opaque
    # translated text regardless, so it should never be treated as an
    # obstacle. RETR_EXTERNAL + fillPoly reproduces the same "ignore
    # interior content" behavior the old lir.lir(polygon) path had (it
    # re-rasterizes the outer contour solid internally too), we just do it
    # ourselves so maximal_rectangle can run on the (cheaper, holes-filled)
    # grid directly instead of going through contour candidate search.
    contours, _hierarchy = cv2.findContours(
        padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.array([0, 0, width, height], dtype=np.int32)

    largest_contour = max(contours, key=cv2.contourArea)

    solid = np.zeros_like(padded)
    cv2.fillPoly(solid, [largest_contour], (255,))

    rect = maximal_rectangle(solid > 0)

    if rect[2] == 0 or rect[3] == 0:
        return np.array([0, 0, width, height], dtype=np.int32)

    p1x, p1y = lir.pt1(rect)
    p2x, p2y = lir.pt2(rect)

    p1x = np.maximum(0, floor((p1x - padding) / scale))
    p1y = np.maximum(0, floor((p1y - padding) / scale))
    p2x = np.maximum(0, ceil(((p2x - padding) + 1) / scale))
    p2y = np.maximum(0, ceil(((p2y - padding) + 1) / scale))
    result = np.array([p1x, p1y, p2x, p2y], dtype=np.int32)
    return result


def get_available_pytorch_devices() -> list[tuple[str, str]]:
    results = [("cpu", "CPU")]

    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            results.append(("cuda", torch.cuda.get_device_name(0)))
        else:
            for i in range(torch.cuda.device_count()):
                results.append((f"cuda:{i}", torch.cuda.get_device_name(i)))

    if torch.backends.mps.is_available():
        results.append(("mps", "Metal Performance Shaders"))

    return results


def get_default_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def inverse_luminance_color(rgb: np.ndarray) -> Vector3u8:
    """
    Return a color with inverted perceived brightness (same hue/saturation).
    Expects `rgb` as np.ndarray of shape (3,) and dtype uint8 in RGB order.
    """
    if rgb.dtype != np.uint8 or rgb.shape != (3,):
        raise ValueError("rgb must be a 1D np.uint8 array of shape (3,) in RGB order")

    # Normalize to [0,1]
    r, g, b = rgb.astype(np.float32) / 255.0

    # Convert to HLS, invert lightness
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
    l = 1.0 - l  # noqa: E741

    # Back to RGB
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(round(x * 255)) for x in (r2, g2, b2)], dtype=np.uint8) # type: ignore


def get_autocast(device: torch.device, enabled=True):
    if not enabled:
        return nullcontext()

    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def standardize_language_code(language: str):
    try:
        return langcodes.Language.get(language).to_tag()
    except langcodes.LanguageTagError: # type: ignore
        pass

    return langcodes.Language.find(language).to_tag()


def get_default_language() -> str:
    return standardize_language_code("en-US")
