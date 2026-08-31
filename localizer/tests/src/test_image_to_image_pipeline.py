"""Behavior-focused tests for ImageToImagePipeline internals and end-to-end flow."""

import numpy as np
import pytest

from comic_localizer.core.constants import DetectionType
from comic_localizer.core.plugin import (
    Cleaner,
    ColorDetectionResult,
    ColorDetector,
    DetectionResult,
    Detector,
    Drawer,
    OCR,
    OcrResult,
    Segmenter,
    Translator,
    TranslatorResult,
)
from comic_localizer.pipelines.image_to_image import ImageToImagePipeline
from comic_localizer.pipelines.image_to_image import FrameSection


BLACK_PIXEL = np.array([0, 0, 0], dtype=np.uint8)
WHITE_PIXEL = np.array([255, 255, 255], dtype=np.uint8)
DRAWN_PIXEL = np.array([200, 200, 200], dtype=np.uint8)
SMALL_FRAME_SHAPE = (6, 6, 3)
ENGLISH_LANGUAGE = "en"
HELLO_TEXT = "hello"
BONJOUR_TEXT = "bonjour"


class StubDetector(Detector):
    def __init__(self, detections):
        self._detections = detections

    async def detect(self, frames):
        return self._detections


class StubSegmenter(Segmenter):
    def __init__(self, segments):
        self._segments = segments

    async def segment(self, frames, detections):
        return self._segments


class StubCleaner(Cleaner):
    def __init__(self, cleaned):
        self._cleaned = cleaned

    async def clean(self, frames, masks, segments, detections):
        return self._cleaned


class StubOCR(OCR):
    def __init__(self, texts):
        self._texts = texts

    async def extract(self, batch):
        return [OcrResult(text=t, language=ENGLISH_LANGUAGE) for t in self._texts]


class StubTranslator(Translator):
    def __init__(self, texts):
        self._texts = texts

    async def translate(self, batch):
        return [
            TranslatorResult(text=t, language=ENGLISH_LANGUAGE) for t in self._texts
        ]


class StubColorDetector(ColorDetector):
    async def detect_color(self, text, cleaned, original):
        return [
            ColorDetectionResult(text_color=np.array([255, 255, 255], dtype=np.uint8))
            for _ in text
        ]


class StubDrawer(Drawer):
    async def draw(self, frames, translations, colors):
        results = []
        for frame in frames:
            drawn = np.full_like(frame, 200, dtype=np.uint8)
            mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
            results.append((drawn, mask))
        return results


def _detection(x1=0, y1=0, x2=2, y2=2):
    return DetectionResult(
        DetectionType.TextInBubble,
        np.array([x1, y1, x2, y2], dtype=np.int32),
        1.0,
    )


@pytest.mark.asyncio
async def test_pipeline_returns_original_frames_when_no_detections():
    """Pipeline should return original objects unchanged when nothing is detected."""
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    pipeline = ImageToImagePipeline(detector=StubDetector([[]]))

    out = await pipeline([frame])

    assert len(out) == 1
    assert out[0] is frame


@pytest.mark.asyncio
async def test_pipeline_cleans_only_detected_regions_when_default_ocr():
    """Default OCR path should stop after cleaning and only replace detected regions."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cleaned = np.full_like(frame, 255, dtype=np.uint8)

    pipeline = ImageToImagePipeline(
        detector=StubDetector([[_detection(1, 1, 3, 3)]]),
        segmenter=StubSegmenter([[]]),
        cleaner=StubCleaner([cleaned]),
        ocr=OCR(),
    )

    out = await pipeline([frame])
    result = out[0]

    assert np.array_equal(result[1, 1], WHITE_PIXEL)
    assert np.array_equal(result[0, 0], BLACK_PIXEL)


@pytest.mark.asyncio
async def test_pipeline_runs_full_translation_draw_flow():
    """When OCR and translation yield content, drawn output should be composited."""
    frame = np.zeros(SMALL_FRAME_SHAPE, dtype=np.uint8)
    cleaned = np.zeros_like(frame)

    pipeline = ImageToImagePipeline(
        detector=StubDetector([[_detection(0, 0, 6, 6)]]),
        segmenter=StubSegmenter([[]]),
        cleaner=StubCleaner([cleaned]),
        ocr=StubOCR([HELLO_TEXT]),
        translator=StubTranslator([BONJOUR_TEXT]),
        color_detector=StubColorDetector(),
        drawer=StubDrawer(),
    )

    out = await pipeline([frame])

    assert len(out) == 1
    assert np.array_equal(out[0], np.full_like(frame, 200, dtype=np.uint8))


@pytest.mark.asyncio
async def test_pipeline_skips_draw_when_ocr_is_empty():
    """Blank OCR text should short-circuit translation and drawing stages."""
    frame = np.zeros(SMALL_FRAME_SHAPE, dtype=np.uint8)

    pipeline = ImageToImagePipeline(
        detector=StubDetector([[_detection(0, 0, 6, 6)]]),
        segmenter=StubSegmenter([[]]),
        cleaner=StubCleaner([frame.copy()]),
        ocr=StubOCR(["   "]),
        translator=StubTranslator(["ignored"]),
        color_detector=StubColorDetector(),
        drawer=StubDrawer(),
    )

    out = await pipeline([frame])
    assert np.array_equal(out[0], frame)


def test_make_mask_fills_polygon_and_returns_empty_without_segments():
    """Mask generation should return all zeros when no segmentation polygons exist."""
    pipeline = ImageToImagePipeline()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    empty = pipeline.make_mask(frame, [])
    assert np.count_nonzero(empty) == 0


def test_clean_frame_using_masks_and_detections_composites_regions():
    """Detection bboxes should define exactly which pixels are replaced by cleaning."""
    pipeline = ImageToImagePipeline()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cleaned = np.full_like(frame, 255, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    out = pipeline.clean_frame_using_masks_and_detections(
        frame,
        cleaned,
        mask,
        [_detection(1, 1, 3, 3)],
    )

    assert np.array_equal(out[1, 1], WHITE_PIXEL)
    assert np.array_equal(out[0, 0], BLACK_PIXEL)


def test_extract_detected_sections_returns_section_with_expected_geometry():
    """Section extraction should preserve source index and crop geometry metadata."""
    pipeline = ImageToImagePipeline()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    cleaned = frame.copy()
    cleaned[2:6, 1:7] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)

    sections = pipeline.extract_detected_sections(
        0,
        frame,
        cleaned,
        mask,
        [_detection(1, 2, 7, 6)],
    )

    assert len(sections) == 1
    section = sections[0]
    assert section.source_index == 0
    assert np.array_equal(section.bbox, np.array([1, 2, 7, 6], dtype=np.int32))
    assert section.section.shape == (4, 6, 3)
    assert section.cleaned_section.shape == (4, 6, 3)
    assert section.mask.shape == (4, 6)
    assert section.draw_section.ndim == 3
    assert section.draw_bbox.shape == (4,)


@pytest.mark.asyncio
async def test_extract_detected_sections_batched_flattens_results_across_frames():
    """Batched section extraction should flatten per-frame sections into one list."""
    pipeline = ImageToImagePipeline()
    frames = [
        np.zeros(SMALL_FRAME_SHAPE, dtype=np.uint8),
        np.zeros(SMALL_FRAME_SHAPE, dtype=np.uint8),
    ]
    cleaned_frames = [f.copy() for f in frames]
    masks = [np.zeros((6, 6), dtype=np.uint8), np.zeros((6, 6), dtype=np.uint8)]
    detections = [
        [_detection(0, 0, 2, 2), _detection(2, 2, 4, 4)],
        [_detection(1, 1, 5, 5)],
    ]

    sections = await pipeline.extract_detected_sections_batched(
        frames,
        cleaned_frames,
        masks,
        detections,
    )

    assert len(sections) == 3
    assert [s.source_index for s in sections].count(0) == 2
    assert [s.source_index for s in sections].count(1) == 1


def test_composite_drawn_sections_overwrites_target_with_masked_drawn_pixels():
    """Compositing should keep cleaned pixels except where drawn mask explicitly applies."""
    pipeline = ImageToImagePipeline()
    source = np.zeros((6, 6, 3), dtype=np.uint8)
    section = FrameSection(
        source_index=0,
        source=source,
        section=source[1:5, 1:5],
        cleaned_section=np.full((4, 4, 3), 100, dtype=np.uint8),
        mask=np.zeros((4, 4), dtype=np.uint8),
        text=np.zeros((4, 4, 3), dtype=np.uint8),
        bbox=np.array([1, 1, 5, 5], dtype=np.int32),
        draw_section=source[1:5, 1:5],
        draw_bbox=np.array([1, 1, 5, 5], dtype=np.int32),
    )

    drawn = np.full((4, 4, 3), 200, dtype=np.uint8)
    drawn_mask = np.zeros((4, 4), dtype=np.uint8)
    drawn_mask[1:3, 1:3] = 255

    pipeline.composite_drawn_sections([(section, drawn, drawn_mask)])

    assert np.array_equal(source[2, 2], DRAWN_PIXEL)
    assert np.array_equal(source[1, 1], np.array([100, 100, 100], dtype=np.uint8))
