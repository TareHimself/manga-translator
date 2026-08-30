"""Tests for GoogleCloudOCR image encoding, word->crop mapping, and extract() contract."""

import base64

import cv2
import numpy as np
import pytest

from comic_localizer.atlas_packing import Placement
from comic_localizer.ocr.google_cloud import (
    GoogleCloudOCR,
    assign_words_to_crops,
    join_words_for_crop,
)


def _make_ocr() -> GoogleCloudOCR:
    return GoogleCloudOCR(api_key="test-key")


def _word(text: str, x0: int, y0: int, x1: int, y1: int) -> dict:
    return {
        "description": text,
        "boundingPoly": {
            "vertices": [
                {"x": x0, "y": y0},
                {"x": x1, "y": y0},
                {"x": x1, "y": y1},
                {"x": x0, "y": y1},
            ]
        },
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_opencv_image_to_b64_preserves_rgb_color_order():
    """Encoding must convert RGB input to BGR before cv2.imencode so the
    resulting PNG round-trips back to the original RGB colors."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:] = (200, 30, 10)  # distinct R, G, B values in RGB order

    encoded_b64 = GoogleCloudOCR.opencv_image_to_b64(image)

    encoded_bytes = base64.b64decode(encoded_b64)
    decoded_bgr = cv2.imdecode(
        np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    assert np.array_equal(decoded_rgb[0, 0], image[0, 0])


def test_assign_words_to_crops_assigns_word_inside_tight_bbox():
    placements = [Placement(id=0, x=0, y=0, width=100, height=50)]
    words = [_word("hi", 10, 10, 30, 20)]

    buckets = assign_words_to_crops(words, placements, padding=20)

    assert buckets == {0: ["hi"]}


def test_assign_words_to_crops_assigns_overflowing_word_via_expanded_rect():
    # crop 0 occupies x:[0,100]; word centroid sits at x=104 - just outside crop
    # 0's tight bbox but inside its padding/2-expanded rect (padding=20 -> +10)
    placements = [
        Placement(id=0, x=0, y=0, width=100, height=50),
        Placement(id=1, x=140, y=0, width=100, height=50),
    ]
    words = [_word("italic", 100, 20, 108, 30)]  # centroid x=104

    buckets = assign_words_to_crops(words, placements, padding=20)

    assert buckets == {0: ["italic"]}


def test_assign_words_to_crops_falls_back_to_nearest_rect_in_gap():
    placements = [
        Placement(id=0, x=0, y=0, width=100, height=50),
        Placement(id=1, x=200, y=0, width=100, height=50),
    ]
    # centroid at x=120 is outside both expanded rects (padding/2=10 -> [-10,110]
    # and [190,310]) but closer to crop 0
    words = [_word("stray", 118, 20, 122, 30)]

    buckets = assign_words_to_crops(words, placements, padding=20)

    assert buckets == {0: ["stray"]}


def test_assign_words_to_crops_preserves_emission_order_per_bucket():
    placements = [
        Placement(id=0, x=0, y=0, width=100, height=50),
        Placement(id=1, x=200, y=0, width=100, height=50),
    ]
    # global emission order interleaves A1, B1, A2 - buckets must not re-sort
    words = [
        _word("A1", 10, 10, 20, 20),
        _word("B1", 210, 10, 220, 20),
        _word("A2", 30, 10, 40, 20),
    ]

    buckets = assign_words_to_crops(words, placements, padding=20)

    assert buckets[0] == ["A1", "A2"]
    assert buckets[1] == ["B1"]
    assert join_words_for_crop(buckets[0]) == "A1 A2"


async def test_extract_gives_zero_size_crop_empty_result_without_consuming_a_composite_slot(
    monkeypatch,
):
    ocr = _make_ocr()
    good_crop = np.zeros((10, 10, 3), dtype=np.uint8)
    zero_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    batch = [good_crop, zero_crop]

    captured_requests = []

    async def fake_post(url, json):
        captured_requests.append(json)
        return _FakeResponse({"responses": [{}]})

    monkeypatch.setattr(ocr.client, "post", fake_post)

    results = await ocr.extract(batch)

    assert len(results) == 2
    assert results[1].text == ""
    assert len(captured_requests) == 1
    # only the valid crop's composite was ever sent
    assert len(captured_requests[0]["requests"]) == 1


async def test_extract_raises_on_response_count_mismatch(monkeypatch):
    ocr = _make_ocr()
    batch = [np.zeros((10, 10, 3), dtype=np.uint8)]

    async def fake_post(url, json):
        return _FakeResponse({"responses": []})

    monkeypatch.setattr(ocr.client, "post", fake_post)

    with pytest.raises(RuntimeError):
        await ocr.extract(batch)
