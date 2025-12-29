from itertools import chain
import cv2
import torch
import numpy as np
import pyphen
import pycountry
from typing import Union, TypeVar
import largestinteriorrectangle as lir
from PIL import Image, ImageFont
from manga_translator.core.typing import Vector4i
T = TypeVar("T")
R = TypeVar("R")


# async def run_in_thread(func, *args, **kwargs):
#     loop = asyncio.get_event_loop()
#     task = asyncio.Future()

#     def run():
#         nonlocal loop
#         nonlocal func
#         nonlocal task

#         result = func(*args, **kwargs)

#         if inspect.isawaitable(result):
#             result = asyncio.run(result)
#         loop.call_soon_threadsafe(task.set_result, result)

#     task_thread = threading.Thread(group=None, daemon=True, target=run)
#     task_thread.start()
#     return await task


# def run_in_thread_decorator(func):
#     async def wrapper(*args, **kwargs):
#         return await run_in_thread(
#             func, *args, **kwargs
#         )  # Comment this out to disable threading

#         result = func(*args, **kwargs)
#         if inspect.isawaitable(result):
#             result = await result
#         return result

#     return wrapper


# async def run_in_thread_async(
#     func: Callable[..., R],
#     batch: Sequence[T],
#     make_args: Callable[[T], tuple[Any, ...]],
# ) -> list[R]:
#     tasks = [asyncio.to_thread(func, *make_args(item)) for item in batch]
#     return await asyncio.gather(*tasks)

class WrappedLine:
    def __init__(self, words: list[str], offset: float,height: float = 0):
        self.words = words
        self.offset = offset
        self.height = height

    def add_word(self,word: str,word_height: float):
        self.words.append(word)
        self.height = max(self.height,word_height)


class WrapResult:
    def __init__(self, lines: list[WrappedLine], bounds: tuple[int, int]):
        self.lines = lines
        self.bounds = bounds


def bbox_to_rect(bbox: tuple[float, float, float, float]):
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


class LayoutCache:
    def __init__(self, font: ImageFont.FreeTypeFont):
        self.font = font
        self.cache = {}

    def get(self, text: str) -> tuple[float, float, float, float]:
        if text in self.cache:
            return self.cache[text]

        self.cache[text] = bbox_to_rect(self.font.getbbox(text))

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

    def filter_out_impossible(self, hyphenations: list[list[str]]):
        # x = list(map(lambda a: list(map(lambda item: [item[0],item[1][2]],a)),hyphenations))
        return filter(lambda hyp: max(hyp,key = lambda item: item[1][2])[1][2] <= self.wrap,hyphenations)

    def get(self, text: str) -> list[list[tuple[str,tuple[float, float, float, float]]]]:
        if text in self.cache:
            return self.cache[text]

        self.cache[text] = list(self.filter_out_impossible(
                [
                    [(text, self.layout_cache.get(text))],
                    *map(self.add_dashes_to_hypenations, self.hyphenator(text))
                ]
            ))

        return self.cache[text]

def has_white(image: np.ndarray):
    # Set RGB values for white
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)

    # Find white pixels within the specified range
    white_pixels = cv2.inRange(image, white_lower, white_upper)

    # Check if any white pixels were found
    return cv2.countNonZero(white_pixels) > 0

def wrap_text_pure(
    text: str, font: ImageFont.FreeTypeFont,line_spacing: float = 2, wrap_width: float = float("inf")
) -> Union[WrapResult, None]:
    layout_cache = LayoutCache(font=font)
    _, _, space_width, _ = layout_cache.get(" ")
    text_list = text.split()
    text_bounds = list(map(lambda a: (a, layout_cache.get(a)), text_list))
    x_offset = 0
    # Text too big to fit on a line
    if any(map(lambda a: a[1][2] > wrap_width, text_bounds)):
        return None

    x_offset = 0
    line_idx = 0
    lines = [WrappedLine([],0)]
    x_bounds = 0
    for word, bbox in text_bounds:
        x_end = x_offset + bbox[2]

        if x_end > wrap_width:
            last_line = lines[-1]
            lines.append(WrappedLine([],last_line.offset + last_line.height + line_spacing))
            line_idx += 1

            x_bounds = max(x_bounds, x_offset)

            x_offset = 0
            x_end = bbox[2]

        lines[line_idx].add_word(word,bbox[3])
        x_offset = min(x_end + space_width, wrap_width)
        x_bounds = max(x_bounds,x_offset)

    last_line = lines[-1]
    return WrapResult(lines, (x_bounds,last_line.offset + last_line.height))


def compute_word_bounds_and_hyphens(
    word: str, font: ImageFont.FreeTypeFont, hyphenator: pyphen.Pyphen
):
    result = [[(word, bbox_to_rect(font.getbbox(word)))]]
    for hyphenated in hyphenator.iterate(word):
        result.append(list(map(lambda a: a, hyphenated)))


def wrap_text_with_hyphenator(
    text: str,
    font: ImageFont.FreeTypeFont,
    hyphenator: pyphen.Pyphen,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
) -> Union[WrapResult, None]:
    layout_cache = LayoutCache(font=font)
    hyphenation_cache = HyphenationCache(
        hyphenator=hyphenator, layout_cache=layout_cache, wrap=wrap_width
    )
    _, _, space_width, _ = layout_cache.get(" ")
    text_list = text.split()
    all_word_versions = list(map(hyphenation_cache.get,text_list))

    # No versions means the word cant fit at this font size
    if any(map(lambda a: len(a) == 0, all_word_versions)):
        return None
    

    # we know one version of the word will fit, we just need to find the version
    def fit_best_version(lines: list[WrappedLine],versions: list[list[tuple[str,tuple[float, float, float, float]]]], x_offset: float, x_bounds: float):
        nonlocal wrap_width
        nonlocal space_width
        nonlocal line_spacing
        line_idx = len(lines) - 1

        selected_version = versions[0]
        version_part_index = 0
        # if we are at a new line we can skip this section
        if x_offset != 0:
            for version in versions:
                word_partial,bbox = version[version_part_index]
                x_end = x_offset + bbox[2]

                if x_end <= wrap_width:
                    lines[line_idx].add_word(word_partial,bbox[3])
                    x_bounds = max(x_bounds,x_end)
                    version_part_index += 1
                    selected_version = version
                    x_offset = x_end + space_width
                    break

    
        if version_part_index < len(selected_version):
            # now we start fitting a new line

            if len(lines[line_idx].words) > 0:
                last_line = lines[-1]
                lines.append(WrappedLine([],last_line.offset + last_line.height + line_spacing))
                line_idx += 1
            
            x_offset = 0

            for version_part in selected_version[version_part_index:]:
                word_partial,bbox = version_part

                x_end = x_offset + bbox[2]

                if x_end > wrap_width:
                    last_line = lines[-1]
                    lines.append(WrappedLine([],last_line.offset + last_line.height + line_spacing))
                    line_idx += 1

                    x_bounds = max(x_bounds, x_offset)

                    x_offset = 0
                    x_end = bbox[2]

                lines[line_idx].add_word(word_partial,bbox[3])
                x_offset = min(x_end + space_width, wrap_width)

                x_bounds = max(x_bounds,x_offset)

        return x_bounds, x_offset


    x_offset = 0
    x_bounds = 0
    lines = [WrappedLine([],0)]

    for versions in all_word_versions:
        new_bounds,new_offset = fit_best_version(lines,versions,x_offset,x_bounds)
        x_bounds = new_bounds
        x_offset = new_offset 

    last_line = lines[-1]
    return WrapResult(lines, (x_bounds,last_line.offset + last_line.height))



def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    wrap_width: float = float("inf"),
    hyphenator: Union[pyphen.Pyphen, None] = None,
    line_spacing: float = 2
) -> Union[WrapResult, None]:
    return (
        wrap_text_pure(text, font, wrap_width,line_spacing)
        if hyphenator is None
        else wrap_text_with_hyphenator(text, font, hyphenator, wrap_width,line_spacing)
    )


class FontFitResult:
    def __init__(self, font_size: float, wrap: WrapResult):
        self.font_size = font_size
        self.wrap = wrap


def find_next_test(min_size, max_size):
    return min_size + ((max_size - min_size) // 2)


def find_best_font_size(
    text: str,
    font_file: str,
    size: tuple[int, int],
    font_size=30,
    min_font_size=10,
    max_font_size=20,
    tolerance=1,
    line_spacing: float = 2,
    hyphenator: Union[pyphen.Pyphen, None] = None,
) -> Union[FontFitResult, None]:
    current_size = min(font_size, max_font_size)
    current_max = max_font_size
    current_min = min_font_size
    best = None
    while True:
        font = ImageFont.truetype(font_file, current_size)
        wrap_result = wrap_text(
            text, font, size[0], hyphenator,line_spacing
        )
        if wrap_result is not None and wrap_result.bounds[1] <= size[1]:
            best = FontFitResult(current_size, wrap_result)
            current_min = current_size
            next_font_size = find_next_test(current_min, current_max)

            if abs(best.font_size - next_font_size) < tolerance:
                break

            current_size = next_font_size

        else:
            current_max = current_size
            next_font_size = find_next_test(current_min, current_max)
            best_font_size = current_size if best is None else best.font_size

            if abs(best_font_size - next_font_size) < tolerance:
                break

            current_size = next_font_size

    return best

def cv2_to_pil(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img: Image) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def ensure_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) > 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    return img.copy()

def draw_area_bbox(section: np.ndarray) -> Vector4i:
    grey = ensure_gray(section)

    height,width = grey.shape[:2]

    ret, thresh = cv2.threshold(grey, 200, 255, 0)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    
    if len(contours) == 0:
        return np.array([0,0,width,height],dtype=np.int32)

    largest_contour = max(contours, key=cv2.contourArea)
    polygon = np.array([largest_contour[:, 0, :]])

    # cv2.imshow("foo",cv2.fillPoly(section.copy(),largest_contour,(255,0,0,255)))
    # cv2.waitKey(0)
    rect = lir.lir(polygon)

    p1x,p1y = lir.pt1(rect)
    p2x,p2y = lir.pt2(rect)

    return np.array([p1x,p1y,p2x,p2y],dtype=np.int32)


def simplify_lang_code(code: str) -> Union[str, None]:
    try:
        lang = pycountry.languages.lookup(code)

        return getattr(lang, "alpha_2", getattr(lang, "alpha_3", None))
    except:
        return None

_languages = list(
        filter(
            lambda a: a[1] is not None,
            list(
                map(
                    lambda a: (
                        getattr(a, "alpha_2", getattr(a, "alpha_3", None)),
                        a.name
                    ),
                    list(pycountry.languages),
                )
            ),
        )
    )
_languages.sort(key=lambda a: a[0].lower())
def get_languages() -> list[tuple[str, str]]:
    return _languages

def get_available_pytorch_devices() -> list[tuple[str,str]]:
    results = [("cpu" ,"CPU")]
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            results.append(("cuda",torch.cuda.get_device_name(0)))
        else:
            for i in range(torch.cuda.device_count()):
                results.append((f"cuda:{i}",torch.cuda.get_device_name(i)))

    if torch.backends.mps.is_available():
        results.append(("mps","Metal Performance Shaders"))

    return results
    

def lang_code_to_name(code: str) -> Union[str, None]:
    try:
        return pycountry.languages.lookup(code).name
    except:
        return None
    
def get_default_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    
    if torch.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")
