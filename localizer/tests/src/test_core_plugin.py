"""Behavior contracts for base plugin types and plugin argument handling."""

import numpy as np
import pytest
import torch

from comic_localizer.core.constants import DetectionType, SegmentationType
from comic_localizer.core.plugin import (
    BasePlugin,
    BooleanPluginArgument,
    Cleaner,
    ColorDetectionResult,
    ColorDetector,
    DetectionResult,
    Detector,
    Drawer,
    IntPluginArgument,
    LanguageStringArgument,
    OCR,
    OcrResult,
    PluginSelectArgumentOption,
    PytorchDevicePluginArgument,
    Segmenter,
    SegmentationResult,
    SelectPluginArgument,
    StringPluginArgument,
    Translator,
    construct_plugin,
)


WHITE_PIXEL = np.array([255, 255, 255], dtype=np.uint8)
BLACK_PIXEL = np.array([0, 0, 0], dtype=np.uint8)
EXAMPLE_PLUGIN_TEXT = "hello"
ENGLISH_LANGUAGE_NAME = "English"
ENGLISH_LANGUAGE_TAG = "en"
FRENCH_LANGUAGE_TAG = "fr-FR"
FAST_OPTION_NAME = "Fast"
FAST_OPTION_VALUE = "fast"
CUDA_ZERO_DEVICE = "cuda:0"
CUDA_DEVICE = "cuda"
UNKNOWN_DEVICE = "unknown"
CPU_DEVICE = "cpu"


class ExamplePlugin(BasePlugin):
    def __init__(self, text="", enabled=False):
        super().__init__()
        self.text = text
        self.enabled = enabled

    @staticmethod
    def get_arguments():
        return [
            StringPluginArgument("text", "Text", "Text value", default="default"),
            BooleanPluginArgument(
                "enabled", "Enabled", "Enable flag", default=False, convert_fn=bool
            ),
        ]


def test_construct_plugin_uses_convert_and_skips_unknown_keys():
    """construct_plugin should convert known keys and ignore unknown keys safely."""
    plugin = construct_plugin(
        ExamplePlugin,
        {"text": EXAMPLE_PLUGIN_TEXT, "enabled": 1, "ignored": "x"},
    )

    assert isinstance(plugin, ExamplePlugin)
    assert plugin.text == EXAMPLE_PLUGIN_TEXT
    assert plugin.enabled is True


def test_plugin_argument_serialization_variants():
    """Argument objects should serialize with expected type ids and option payloads."""
    select = SelectPluginArgument(
        "mode",
        "Mode",
        "How to run",
        [PluginSelectArgumentOption(FAST_OPTION_NAME, FAST_OPTION_VALUE)],
        default=FAST_OPTION_VALUE,
    )

    int_arg = IntPluginArgument("count", "Count", "How many", default=3)
    bool_arg = BooleanPluginArgument("flag", "Flag", "Enabled", default=True)

    assert select.get()["options"] == [
        {"name": FAST_OPTION_NAME, "value": FAST_OPTION_VALUE}
    ]
    assert int_arg.get()["type"] == 2
    assert bool_arg.get()["type"] == 3


def test_language_string_argument_converts_human_name_and_tag():
    """Language argument should normalize both language names and already-tagged values."""
    arg = LanguageStringArgument("language", "Language", "Target language")

    assert arg.convert_lang(ENGLISH_LANGUAGE_NAME) == ENGLISH_LANGUAGE_TAG
    assert arg.convert_lang(FRENCH_LANGUAGE_TAG) == FRENCH_LANGUAGE_TAG


def test_pytorch_device_argument_conversion(monkeypatch):
    """Device conversion should normalize cuda:0 and fall back to first available device."""
    monkeypatch.setattr(
        PytorchDevicePluginArgument,
        "available_devices_flat",
        [CPU_DEVICE, CUDA_DEVICE, "mps"],
    )

    arg = PytorchDevicePluginArgument("device", "Device")

    assert arg.convert_to_torch_device(CUDA_ZERO_DEVICE) == torch.device(CUDA_DEVICE)
    assert arg.convert_to_torch_device(UNKNOWN_DEVICE) == torch.device(CPU_DEVICE)


@pytest.mark.asyncio
async def test_base_ocr_translator_detector_and_segmenter_contracts():
    """Default OCR/translator/detector/segmenter should satisfy base contract invariants."""
    ocr = OCR()
    translator = Translator()
    detector = Detector()
    segmenter = Segmenter()

    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

    ocr_results = await ocr(frames)
    assert len(ocr_results) == 1
    assert isinstance(ocr_results[0], OcrResult)
    assert ocr_results[0].text == ""

    translated = await translator(ocr_results)
    assert translated[0].text == ""

    detections = await detector(frames)
    segments = await segmenter(frames, detections)

    assert detections == [[]]
    assert segments == [[]]


@pytest.mark.asyncio
async def test_base_cleaner_drawer_color_detector_contracts():
    """Default cleaner/drawer/color detector should produce structurally valid outputs."""
    cleaner = Cleaner()
    drawer = Drawer()
    color_detector = ColorDetector()

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    cleaned = await cleaner([frame], [mask], [[]], [[]])
    assert np.array_equal(cleaned[0], frame)
    assert cleaned[0] is not frame

    colors = await color_detector([frame], [frame], [frame])
    assert len(colors) == 1
    assert isinstance(colors[0], ColorDetectionResult)

    drawn = await drawer(
        [frame],
        [],
        colors,
    )
    assert len(drawn) == 1
    assert len(drawn[0]) == 2


def test_color_detection_result_defaults_to_inverse_outline_color():
    """ColorDetectionResult should derive outline color from inverse luminance by default."""
    text_color = WHITE_PIXEL
    result = ColorDetectionResult(text_color=text_color)

    assert result.outline_size == 0
    assert np.array_equal(result.outline_color, BLACK_PIXEL)


def test_detection_and_segmentation_result_objects_store_values():
    """Data containers should preserve constructor values without coercion."""
    detection = DetectionResult(
        DetectionType.TextInBubble,
        np.array([1, 2, 3, 4], dtype=np.int32),
        0.99,
    )
    segmentation = SegmentationResult(
        SegmentationType.Text,
        [np.array([1.0, 2.0], dtype=np.float32)],
        0.9,
    )

    assert detection.type == DetectionType.TextInBubble
    assert detection.confidence == 0.99
    assert np.array_equal(detection.bbox, np.array([1, 2, 3, 4], dtype=np.int32))

    assert segmentation.type == SegmentationType.Text
    assert len(segmentation.points) == 1
    assert segmentation.confidence == 0.9
