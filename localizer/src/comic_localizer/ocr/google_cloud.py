import asyncio
import base64
from dataclasses import dataclass
from more_itertools import chunked
import numpy as np
from comic_localizer.core.plugin import (
    OCR,
    IntPluginArgument,
    OcrResult,
    PluginArgument,
    StringPluginArgument,
)
from comic_localizer.atlas_packing import AtlasLayout, Placement, pack_into_atlases
from comic_localizer.utils import get_default_language, lingua_lang_to_lang_code
from lingua import LanguageDetectorBuilder
import cv2
import httpx

# Cloud Vision bills per image regardless of resolution/content, so crops are
# packed into shared composites before being sent - see atlas_packing.py.
DEFAULT_BIN_SIZE = 1536  # px, square composite target (~2.36MP)
DEFAULT_PADDING = 32  # px gap enforced between every pair of packed crops
DEFAULT_BYTE_CAP_BYTES = 7 * 1024 * 1024  # raw-pixel estimate cap, safety margin
# under Google's 20MB/image cap and the ~10MB inline-base64 JSON request cap


@dataclass
class PackedComposite:
    image: np.ndarray
    encoded: str
    placements: list[Placement]


def filter_valid_crops(batch: list[np.ndarray]) -> tuple[list[int], list[np.ndarray]]:
    """Drops zero width/height crops - callers get an empty OcrResult() for these
    without spending a composite slot or an API call."""
    indices: list[int] = []
    crops: list[np.ndarray] = []
    for i, crop in enumerate(batch):
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            indices.append(i)
            crops.append(crop)
    return indices, crops


def render_composite(crops_by_index: dict[int, np.ndarray], layout: AtlasLayout) -> np.ndarray:
    canvas = np.full((layout.height, layout.width, 3), 255, dtype=np.uint8)
    for placement in layout.placements:
        crop = crops_by_index[placement.id]
        canvas[
            placement.y : placement.y + placement.height,
            placement.x : placement.x + placement.width,
        ] = crop
    return canvas


def build_composites(
    indices: list[int],
    crops: list[np.ndarray],
    bin_size: int,
    padding: int,
    byte_cap: int,
) -> list[PackedComposite]:
    crops_by_index = dict(zip(indices, crops))
    items = [(i, c.shape[1], c.shape[0]) for i, c in zip(indices, crops)]
    layouts = pack_into_atlases(items, bin_size, padding, byte_cap)

    composites: list[PackedComposite] = []
    for layout in layouts:
        image = render_composite(crops_by_index, layout)
        encoded = GoogleCloudOCR.opencv_image_to_b64(image)
        composites.append(
            PackedComposite(image=image, encoded=encoded, placements=layout.placements)
        )
    return composites


def _bbox_from_vertices(vertices: list[dict]) -> tuple[float, float, float, float]:
    # Vision omits zero-valued x/y keys entirely, so default to 0
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(vertices: list[dict]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = _bbox_from_vertices(vertices)
    return (min_x + max_x) / 2, (min_y + max_y) / 2


def _expand_rect(placement: Placement, amount: float) -> tuple[float, float, float, float]:
    return (
        placement.x - amount,
        placement.y - amount,
        placement.x + placement.width + amount,
        placement.y + placement.height + amount,
    )


def _rect_contains_point(rect: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    x0, y0, x1, y1 = rect
    px, py = point
    return x0 <= px <= x1 and y0 <= py <= y1


def _distance_point_to_rect(rect: tuple[float, float, float, float], point: tuple[float, float]) -> float:
    x0, y0, x1, y1 = rect
    px, py = point
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return (dx**2 + dy**2) ** 0.5


def assign_words_to_crops(
    word_annotations: list[dict], placements: list[Placement], padding: int
) -> dict[int, list[str]]:
    """Buckets Vision's word-level annotations by which packed crop's rect
    (expanded by padding/2 to tolerate glyph overflow) contains its centroid,
    falling back to the nearest rect for words that land in the padding gap."""
    expanded = [(p, _expand_rect(p, padding / 2)) for p in placements]
    buckets: dict[int, list[str]] = {}

    for word in word_annotations:
        centroid = _centroid(word["boundingPoly"]["vertices"])
        match = next((p for p, rect in expanded if _rect_contains_point(rect, centroid)), None)
        if match is None:
            match = min(
                placements,
                key=lambda p: _distance_point_to_rect(
                    (p.x, p.y, p.x + p.width, p.y + p.height), centroid
                ),
            )
        buckets.setdefault(match.id, []).append(word["description"])

    return buckets


def join_words_for_crop(words: list[str]) -> str:
    # Trusts Vision's own word emission order rather than re-sorting geometrically.
    # Risky for vertical CJK columns, where Vision's whole-composite reading-order
    # heuristic can interleave across adjacent crops - verify against real vertical
    # text before relying on this in production.
    return " ".join(words)


def build_ocr_results_into(
    results: list[OcrResult],
    composites: list[PackedComposite],
    vision_responses: list[dict],
    padding: int,
) -> None:
    for composite, vision_response in zip(composites, vision_responses):
        annotations = vision_response.get("textAnnotations")
        if not annotations:
            continue

        aggregate_locale = annotations[0].get("locale")
        buckets = assign_words_to_crops(annotations[1:], composite.placements, padding)

        for placement in composite.placements:
            words = buckets.get(placement.id)
            if not words:
                continue
            results[placement.id] = OcrResult(
                text=join_words_for_crop(words),
                language=aggregate_locale or get_default_language(),
            )


class GoogleCloudOCR(OCR):
    """Uses google cloud to perform ocr , requires an API key"""

    def __init__(
        self,
        api_key: str,
        bin_size: int = DEFAULT_BIN_SIZE,
        padding: int = DEFAULT_PADDING,
        byte_cap: int = DEFAULT_BYTE_CAP_BYTES,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.bin_size = bin_size
        self.padding = padding
        self.byte_cap = byte_cap
        self.client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            base_url="https://vision.googleapis.com/v1",
        )
        self.language_detector = LanguageDetectorBuilder.from_all_spoken_languages().build()

    @staticmethod
    def opencv_image_to_b64(image: np.ndarray):
        success, encoded_bytes = cv2.imencode(
            ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise RuntimeError("Failed to encode image")

        # google does not want the data.../... part
        return base64.b64encode(encoded_bytes).decode("utf-8")

    async def _refine_languages(self, results: list[OcrResult]) -> None:
        # Word-level annotations don't carry a reliable locale, only the whole
        # composite's aggregate does - re-detect per-crop language locally
        # (free, non-billed) for better precision than the aggregate fallback.
        candidates = [(i, r.text) for i, r in enumerate(results) if r.text.strip()]
        if not candidates:
            return

        texts = [text for _, text in candidates]
        detected = await asyncio.to_thread(
            self.language_detector.detect_languages_in_parallel_of, texts
        )

        for (index, _), language in zip(candidates, detected):
            if language is not None:
                results[index].language = lingua_lang_to_lang_code(language)

    async def extract(self, batch: list[np.ndarray]):
        valid_indices, valid_crops = filter_valid_crops(batch)
        ocr_results: list[OcrResult] = [OcrResult() for _ in batch]
        if not valid_crops:
            return ocr_results

        composites = await asyncio.to_thread(
            build_composites,
            valid_indices,
            valid_crops,
            self.bin_size,
            self.padding,
            self.byte_cap,
        )

        # 16 is the cap set by google https://docs.cloud.google.com/vision/quotas
        for group in chunked(composites, 16):
            response = await self.client.post(
                "/images:annotate",
                json={
                    "requests": [
                        {
                            "image": {"content": composite.encoded},
                            "features": [{"type": "TEXT_DETECTION"}],
                        }
                        for composite in group
                    ]
                },
            )

            response.raise_for_status()

            result = response.json()

            if len(result["responses"]) != len(group):
                raise RuntimeError(
                    f"GoogleCloudOCR: sent {len(group)} composite images but got back "
                    f"{len(result['responses'])} responses"
                )

            build_ocr_results_into(ocr_results, group, result["responses"], self.padding)

        await self._refine_languages(ocr_results)

        return ocr_results

    @staticmethod
    def get_name() -> str:
        return "Google Cloud OCR"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="api_key", name="API Key", description="Google cloud Vision API key"
            ),
            IntPluginArgument(
                id="bin_size",
                name="Composite Size",
                description="Target width/height (px) of each packed composite image sent to Vision",
                default=DEFAULT_BIN_SIZE,
            ),
            IntPluginArgument(
                id="padding",
                name="Crop Padding",
                description="Gap (px) enforced between crops packed into the same composite",
                default=DEFAULT_PADDING,
            ),
            IntPluginArgument(
                id="byte_cap",
                name="Composite Byte Cap",
                description="Max estimated raw bytes per composite before it's split further",
                default=DEFAULT_BYTE_CAP_BYTES,
            ),
        ]
