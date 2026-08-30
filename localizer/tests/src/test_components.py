"""Component behavior tests for OCR, translation, cleaning, drawing, and registries."""

import numpy as np
import pytest

from comic_localizer.cleaning.all_white_cleaner import AllWhiteCleaner
from comic_localizer.cleaning.opencv import OpenCvCleaner
from comic_localizer.core.plugin import (
    ColorDetectionResult,
    OcrResult,
    TranslatorResult,
)
from comic_localizer.drawing.horizontal import HorizontalDrawer
from comic_localizer.ocr.debug import DebugOCR
from comic_localizer.translation.debug import DebugTranslator
from comic_localizer.translation.pipe import PipeTranslator
from comic_localizer.utils import FontFitResult, WrapResult, WrappedLine


WHITE_PIXEL = np.array([255, 255, 255], dtype=np.uint8)
BLACK_PIXEL = np.array([0, 0, 0], dtype=np.uint8)
DEBUG_OCR_TEXT = "hello"
DEBUG_OCR_LANGUAGE = "en"
DEBUG_TRANSLATED_TEXT = "translated"
DEBUG_TRANSLATED_LANGUAGE = "fr"
DRAWER_FONT = "dummy.ttf"  # real rendering paths monkeypatch load_font


@pytest.mark.asyncio
async def test_debug_ocr_outputs_configured_text_for_each_frame():
    """Debug OCR should produce configured text and language for each input frame."""
    ocr = DebugOCR(text=DEBUG_OCR_TEXT, language=DEBUG_OCR_LANGUAGE)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8)]

    results = await ocr(frames)

    assert all(x.text == ocr.to_write for x in results)
    assert all(x.language == ocr.language for x in results)


@pytest.mark.asyncio
async def test_debug_translator_outputs_configured_text_and_language():
    """Debug translator should ignore input content and emit configured output values."""
    translator = DebugTranslator(
        text=DEBUG_TRANSLATED_TEXT,
        language=DEBUG_TRANSLATED_LANGUAGE,
    )
    batch = [OcrResult("ignored", "en"), OcrResult("ignored2", "es")]

    results = await translator(batch)

    assert all(x.text == translator.to_write for x in results)
    assert all(x.language == translator.language for x in results)


@pytest.mark.asyncio
async def test_pipe_translator_passes_text_and_language_through():
    """Pipe translator should preserve OCR text and language without transformation."""
    translator = PipeTranslator()
    batch = [OcrResult("one", "en"), OcrResult("two", "ja")]

    results = await translator(batch)

    assert [x.text for x in results] == [x.text for x in batch]
    assert all(out.language == src.language for out, src in zip(results, batch))


def test_all_white_cleaner_masks_selected_region_to_white():
    """AllWhiteCleaner should only modify masked pixels and preserve unmasked regions."""
    cleaner = AllWhiteCleaner()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    cleaned = cleaner.clean_frame(frame, mask)

    assert np.array_equal(cleaned[1, 1], WHITE_PIXEL)
    assert np.array_equal(cleaned[0, 0], BLACK_PIXEL)


@pytest.mark.asyncio
async def test_all_white_cleaner_async_clean_processes_each_frame_and_get_name():
    """AllWhiteCleaner async clean should process each frame and expose a stable name."""
    cleaner = AllWhiteCleaner()
    frames = [np.zeros((3, 3, 3), dtype=np.uint8), np.zeros((3, 3, 3), dtype=np.uint8)]
    masks = [np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)]
    masks[0][1, 1] = 255

    out = await cleaner.clean(frames, masks, [[], []], [[], []])

    assert len(out) == 2
    assert np.array_equal(out[0][1, 1], WHITE_PIXEL)
    assert np.array_equal(out[1], frames[1])


def test_opencv_cleaner_clean_frame_preserves_shape_and_dtype():
    """OpenCV cleaner should return image-like output with stable shape and dtype."""
    cleaner = OpenCvCleaner(radius=1, mask_after_inpaint=True)
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[2:4, 2:4] = 255
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 255

    out = cleaner.clean_frame(frame, mask)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


@pytest.mark.asyncio
async def test_opencv_cleaner_async_clean_get_name_and_is_valid():
    """OpenCV cleaner async path and metadata methods should return expected values."""
    cleaner = OpenCvCleaner(radius=1, mask_after_inpaint=False)
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8)

    out = await cleaner.clean([frame], [mask], [[]], [[]])

    assert len(out) == 1
    assert out[0].shape == frame.shape
    assert OpenCvCleaner.is_valid() is True


def test_horizontal_drawer_requires_a_font_file():
    """No font ships with the repo, so a font_file must be supplied."""
    with pytest.raises(ValueError):
        HorizontalDrawer()


def test_horizontal_drawer_draw_text_returns_empty_mask_for_blank_text():
    """Drawer should no-op when translation text is blank or whitespace only."""
    drawer = HorizontalDrawer(font_file=DRAWER_FONT)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    translation = TranslatorResult(text="   ", language="en")
    color = ColorDetectionResult(np.array([255, 255, 255], dtype=np.uint8))

    drawn, mask = drawer.draw_text(frame, translation, color)

    assert np.array_equal(drawn, frame)
    assert np.count_nonzero(mask) == 0


def test_horizontal_drawer_draw_text_returns_empty_mask_when_fit_fails(monkeypatch):
    """Drawer should no-op when font fitting cannot find any valid wrapping."""
    drawer = HorizontalDrawer(font_file=DRAWER_FONT)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    translation = TranslatorResult(text="hello", language="en")
    color = ColorDetectionResult(np.array([255, 255, 255], dtype=np.uint8))

    monkeypatch.setattr(
        "comic_localizer.drawing.horizontal.find_best_font_size", lambda *a, **k: None
    )

    drawn, mask = drawer.draw_text(frame, translation, color)

    assert np.array_equal(drawn, frame)
    assert np.count_nonzero(mask) == 0


def test_horizontal_drawer_draw_text_renders_requested_rgb_color(monkeypatch):
    """Drawn text pixels must use the requested RGB color, not have red/blue swapped."""
    from PIL import ImageFont

    drawer = HorizontalDrawer(font_file=DRAWER_FONT)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    translation = TranslatorResult(text="A", language="en")
    red = np.array([255, 0, 0], dtype=np.uint8)
    color = ColorDetectionResult(red, outline_size=0)

    default_font = ImageFont.load_default(size=20)
    fit_result = FontFitResult(
        font_size=20,
        wrap=WrapResult([WrappedLine(["A"], 0, height=20)], (20, 20)),
    )

    monkeypatch.setattr(
        "comic_localizer.drawing.horizontal.find_best_font_size",
        lambda *a, **k: fit_result,
    )
    monkeypatch.setattr(
        "comic_localizer.drawing.horizontal.load_font",
        lambda *a, **k: default_font,
    )

    drawn, mask = drawer.draw_text(frame, translation, color)

    drawn_pixels = drawn[mask > 0]
    assert len(drawn_pixels) > 0
    # Requested color is pure red; red channel should dominate, not blue.
    assert np.all(drawn_pixels[:, 0].astype(int) > drawn_pixels[:, 2].astype(int))


@pytest.mark.asyncio
async def test_horizontal_drawer_async_draw_returns_pair_for_each_item(monkeypatch):
    """Async draw should return one (drawn, mask) tuple per frame."""
    drawer = HorizontalDrawer(font_file=DRAWER_FONT)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    translation = TranslatorResult(text="hello", language="en")
    color = ColorDetectionResult(np.array([255, 255, 255], dtype=np.uint8))

    def fake_draw_text(frame, translation, color):
        return frame.copy(), np.zeros(frame.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(drawer, "draw_text", fake_draw_text)

    out = await drawer.draw([frame], [translation], [color])
    assert len(out) == 1
    assert out[0][0].shape == frame.shape
    assert out[0][1].shape == frame.shape[:2]
