"""Text-mask segmenter backed by the comic-localizer-text-masking model.

Runs a small U-Net on each detector crop and turns the predicted per-pixel text
mask into polygons. Only ``torch`` and ``huggingface_hub`` are needed: the model
is a self-contained TorchScript file (``model.pt``) with ``/255`` + ImageNet
normalisation + sigmoid baked into the graph, so it takes letterboxed uint8 RGB
and returns a probability map.

    https://huggingface.co/TareHimself/comic-text-mask
"""

import asyncio
import json
from importlib.util import find_spec

import cv2
import numpy as np
import torch

from comic_localizer.core.constants import SegmentationType
from comic_localizer.core.plugin import (
    DetectionResult,
    IntPluginArgument,
    PluginArgument,
    PytorchDevicePluginArgument,
    SegmentationResult,
    Segmenter,
    StringPluginArgument,
)
from comic_localizer.utils import get_default_torch_device

_DEFAULT_REPO = "TareHimself/comic-text-mask"


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Aspect-preserving resize + centre-pad to a ``size`` square (matches the
    training letterbox). Returns the canvas and the ``(x, y, w, h)`` box the
    image was pasted into."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    canvas = np.zeros((size, size, 3), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA
    )
    return canvas, (ox, oy, nw, nh)


class TextMaskSegmenter(Segmenter):
    def __init__(
        self,
        repo: str = _DEFAULT_REPO,
        revision: str = "",
        device: torch.device = get_default_torch_device(),
        threshold: float = 0.5,
        pad: float = 0.1,
        min_area: int = 12,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        from huggingface_hub import hf_hub_download

        rev = revision or None
        with open(
            hf_hub_download(repo, "tm_meta.json", revision=rev), encoding="utf-8"
        ) as f:
            meta = json.load(f)
        self.imgsz = int(meta["imgsz"])
        self.threshold = threshold
        self.pad = pad
        self.min_area = min_area
        self.batch_size = batch_size
        self.device = device
        self.model = torch.jit.load(
            hf_hub_download(repo, "model.pt", revision=rev), map_location=device
        ).eval()

    def _crop_box(self, box, w: int, h: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (int(v) for v in box)
        px, py = int((x2 - x1) * self.pad), int((y2 - y1) * self.pad)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None

    @torch.inference_mode()
    def _probs(self, canvases: list[np.ndarray]) -> np.ndarray:
        """Run the model over every letterboxed crop in one (chunked) pass."""
        batch = torch.from_numpy(np.stack(canvases)).permute(0, 3, 1, 2)
        chunks = [
            self.model(batch[i : i + self.batch_size].to(self.device))[:, 0]
            .float()
            .cpu()
            .numpy()
            for i in range(0, len(batch), self.batch_size)
        ]
        return np.concatenate(chunks)

    def predict(
        self,
        frames: list[np.ndarray],
        detections: list[list[DetectionResult]],
    ) -> list[list[SegmentationResult]]:
        out: list[list[SegmentationResult]] = [[] for _ in frames]

        canvases: list[np.ndarray] = []
        jobs: list[tuple] = []  # (frame_idx, crop_box, letterbox_box, detection)
        for fi, (frame, dets) in enumerate(zip(frames, detections)):
            h, w = frame.shape[:2]
            for det in dets:
                box = self._crop_box(det.bbox, w, h)
                if box is None:
                    continue
                canvas, lb = _letterbox(frame[box[1] : box[3], box[0] : box[2]], self.imgsz)
                canvases.append(canvas)
                jobs.append((fi, box, lb, det))

        if not canvases:
            return out

        for (fi, (x1, y1, x2, y2), (ox, oy, nw, nh), det), prob in zip(
            jobs, self._probs(canvases)
        ):
            p = cv2.resize(
                prob[oy : oy + nh, ox : ox + nw],
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = (p > self.threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if cv2.contourArea(c) < self.min_area:
                    continue
                poly = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) + (x1, y1)
                if len(poly) >= 3:
                    out[fi].append(
                        SegmentationResult(
                            SegmentationType.Text, poly.astype(int), det.confidence
                        )
                    )
        return out

    async def segment(
        self,
        frames: list[np.ndarray],
        detections: list[list[DetectionResult]],
    ) -> list[list[SegmentationResult]]:
        return await asyncio.to_thread(self.predict, frames, detections)

    @staticmethod
    def is_valid() -> bool:
        return find_spec("huggingface_hub") is not None

    @staticmethod
    def get_name() -> str:
        return "Text Mask"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                "repo", "Model Repo", "HuggingFace model repo id", _DEFAULT_REPO
            ),
            StringPluginArgument(
                "revision", "Revision", "Tag / branch / commit (blank = latest)", "v3"
            ),
            PytorchDevicePluginArgument("device", "Device"),
            IntPluginArgument(
                "batch_size", "Batch Size", "Crops per forward pass", 16
            ),
        ]
