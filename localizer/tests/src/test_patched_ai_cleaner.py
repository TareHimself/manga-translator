"""Tests for PatchedAiCleaner mask polarity / compositing and per-cleaner model loading."""

import numpy as np
import pytest
import torch

from comic_localizer.cleaning.patched_ai_cleaner import PatchedAiCleaner
from comic_localizer.cleaning.deepfillv2 import DeepFillV2Cleaner
from comic_localizer.cleaning.lama import LamaCleaner
from comic_localizer.core.constants import DetectionType, SegmentationType
from comic_localizer.core.plugin import DetectionResult, SegmentationResult


class _ConstantInpaintModel:
    """Stand-in for a torch.jit inpainting model: ignores input, returns a
    constant-valued tensor of the same shape so output can be asserted on."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, batch: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return torch.full_like(batch, self.value)


def _make_cleaner(monkeypatch, model=None, **kwargs) -> PatchedAiCleaner:
    monkeypatch.setattr(PatchedAiCleaner, "load_model", lambda self, device: model)
    return PatchedAiCleaner(**kwargs)


def test_extract_patches_mask_background_is_zero_not_one(monkeypatch):
    """The mask background outside the segment polygon must be 0 (keep), not 1,
    since process_masks_batched divides by 255 and 1 would leak a tiny nonzero
    signal into every 'keep this pixel' region fed to the inpainting model."""
    cleaner = _make_cleaner(monkeypatch)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    points = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
    segment = SegmentationResult(SegmentationType.Text, points, confidence=1.0)

    patches = cleaner.extract_patches([frame], [[segment]])

    assert len(patches) == 1
    mask = patches[0].mask
    assert mask[0, 0] == 0
    assert mask[9, 9] == 255


def test_clean_sync_without_patching_preserves_caller_frames_and_composites_correctly(
    monkeypatch,
):
    """With inpaint_patches=False, clean_sync must not mutate the caller's
    `frames` list, and must composite cleaned content inside detection boxes
    with the original content outside them (not the reverse)."""
    model = _ConstantInpaintModel(0.5)
    cleaner = _make_cleaner(monkeypatch, model=model, inpaint_patches=False)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    original_frame = frame.copy()
    mask = np.zeros((16, 16), dtype=np.uint8)
    detection = DetectionResult(DetectionType.TextInBubble, (4, 4, 12, 12), 1.0)

    result = cleaner.clean_sync([frame], [mask], [[]], [[detection]])

    # Caller's original frame array must be untouched.
    assert np.array_equal(frame, original_frame)

    out = result[0]
    # Inside the detection box: cleaned (model constant ~127), not the original black.
    assert out[6, 6, 0] == pytest.approx(127, abs=2)
    # Outside the detection box: original content (black), not the cleaned constant.
    assert np.array_equal(out[0, 0], np.array([0, 0, 0], dtype=np.uint8))


def test_base_class_load_model_must_be_overridden():
    """PatchedAiCleaner is abstract in practice: constructing it (which calls
    load_model) without a subclass override raises."""
    with pytest.raises(NotImplementedError):
        PatchedAiCleaner()


def test_base_arguments_no_longer_expose_a_model_path():
    ids = {a.id for a in PatchedAiCleaner.get_arguments()}
    assert "model_path" not in ids
    assert {"inpaint_patches", "patch_padding", "device"} <= ids


def test_deepfillv2_loads_from_its_model_path(monkeypatch):
    seen = {}
    sentinel = object()

    def fake_load(path, map_location=None):
        seen["path"], seen["map_location"] = path, map_location

        class _M:
            def eval(self_inner):
                return sentinel

        return _M()

    monkeypatch.setattr(torch.jit, "load", fake_load)

    cleaner = DeepFillV2Cleaner(model_path="/models/deepfill.pt", device=torch.device("cpu"))

    assert cleaner.model is sentinel
    assert seen["path"] == "/models/deepfill.pt"
    assert seen["map_location"] == torch.device("cpu")


def test_deepfillv2_exposes_model_path_plus_the_base_arguments():
    ids = [a.id for a in DeepFillV2Cleaner.get_arguments()]
    assert ids[0] == "model_path"
    assert {"inpaint_patches", "patch_padding", "device"} <= set(ids)


def test_lama_downloads_its_model_from_the_hub(monkeypatch):
    seen = {}
    sentinel = object()

    monkeypatch.setattr(
        "comic_localizer.cleaning.lama.hf_hub_download",
        lambda repo, filename, revision=None: seen.update(
            repo=repo, filename=filename, revision=revision
        )
        or "/cache/anime_manga_lama.pt",
    )

    def fake_load(path, map_location=None):
        seen["path"] = path

        class _M:
            def eval(self_inner):
                return sentinel

        return _M()

    monkeypatch.setattr(torch.jit, "load", fake_load)

    cleaner = LamaCleaner(device=torch.device("cpu"))

    assert cleaner.model is sentinel
    assert seen["repo"] == "TareHimself/AnimeMangaInpainting-torchscript"
    assert seen["filename"] == "anime_manga_lama.pt"
    assert seen["revision"] == "v1"
    assert seen["path"] == "/cache/anime_manga_lama.pt"


def test_lama_takes_no_model_path_argument():
    ids = {a.id for a in LamaCleaner.get_arguments()}
    assert "model_path" not in ids
