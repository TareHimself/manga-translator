# Adapted from https://github.com/nipponjo/deepfillv2-pytorch
from manga_translator.core.plugin import (
    Cleaner,
    PluginArgument,
    PytorchDevicePluginSelectArgument,
    SegmentationResult,
    DetectionResult,
    IntPluginArgument,
    BooleanPluginArgument,
    StringPluginArgument,
)
import numpy as np
import cv2
import asyncio
import torch
import torch.nn.functional as F
from manga_translator.utils import get_default_torch_device
import torchvision.transforms.v2 as T


class _DeepFillV2ImagePatch:
    def __init__(
        self,
        source: np.ndarray,
        patch: np.ndarray,
        mask: np.ndarray,
        offset: np.ndarray,
        actual_size: np.ndarray,
        offset_padding: np.ndarray,
    ):
        self.source = source
        self.patch = patch
        self.mask = mask
        self.offset = offset
        self.actual_size = actual_size
        self.offset_padding = offset_padding


class DeepFillV2Cleaner(Cleaner):
    """Cleans using Free-Form Image Inpainting with Gated Convolution https://arxiv.org/abs/1806.03589"""

    def __init__(
        self,
        model_path: str,
        inpaint_patches=True,
        patch_padding=10,
        device: torch.device = get_default_torch_device(),
    ) -> None:
        super().__init__()
        self.device = device
        self.model = torch.jit.freeze(torch.jit.load(model_path, map_location=device).eval())
        self.inpaint_patches = inpaint_patches
        self.patch_padding = np.array([patch_padding, patch_padding], dtype=np.int32)
        self.zero_max = np.zeros((2), dtype=np.int32)

    def process_input(self, x: np.ndarray):
        t = torch.from_numpy(x)
        if len(t.shape) == 2:
            t = t.unsqueeze(0)
        else:
            t = t.permute(2, 0, 1).flip(0)
        c, h, w = t.shape

        h_padding = 0 if (h % 8) == 0 else (8 * (h // 8) + 8) - h
        w_padding = 0 if (w % 8) == 0 else (8 * (w // 8) + 8) - w

        padded = F.pad(t, (0, w_padding, 0, h_padding), mode="replicate")
        return padded / 255.0

    def process_output(self, x: torch.Tensor):
        x = (
            (x * 255).byte().flip(0).permute(1, 2, 0)
        )  # Flip back to BGR then back to numpy
        x = x.numpy()
        return x

    def extract_patches(
        self, frames: list[np.ndarray], segments: list[list[SegmentationResult]] = []
    ) -> list[_DeepFillV2ImagePatch]:
        patches: list[_DeepFillV2ImagePatch] = []

        for frame, frame_segments in zip(frames, segments):
            h, w = frame.shape[:2]
            size_vec = np.array([w, h], dtype=np.int32)
            for segment in frame_segments:

                actual_p1 = np.array(np.minimum.reduce(segment.points))
                actual_p2 = np.array(np.maximum.reduce(segment.points))
                p1 = np.maximum(actual_p1 - self.patch_padding, self.zero_max)
                p2 = np.minimum(actual_p2 + self.patch_padding, size_vec)
                final_offfset_padding = actual_p1 - p1
                actual_size = actual_p2 - actual_p1

                frame_patch = frame[p1[1] : p2[1], p1[0] : p2[0]]
                mask_patch_size = p2 - p1
                mask_patch = np.ones(
                    (mask_patch_size[1], mask_patch_size[0]), dtype=frame.dtype
                )
                cv2.fillPoly(
                    mask_patch,
                    [np.array(list(map(lambda a: a - p1, segment.points)))],
                    (255, 255, 255),
                )

                patches.append(
                    _DeepFillV2ImagePatch(
                        frame,
                        frame_patch,
                        mask_patch,
                        actual_p1,
                        actual_size,
                        final_offfset_padding,
                    )
                )

        return patches

    def clean_patches(self, patches: list[_DeepFillV2ImagePatch]):
        with torch.inference_mode():
            # we can do some kind of size based grouping to batch here
            for patch in patches:
                input_tensor: torch.Tensor = self.process_input(patch.patch).to(
                    self.device
                )  # BGR to RGB
                mask_tensor: torch.Tensor = self.process_input(patch.mask).to(
                    self.device
                )
                result: torch.Tensor = self.model(
                    input_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)
                )
                out_image: np.ndarray = self.process_output(result[0].cpu())
                h, w = patch.patch.shape[:2]
                section = out_image[0:h, 0:w].copy()
                patch.source[
                    patch.offset[1] : patch.offset[1] + patch.actual_size[1],
                    patch.offset[0] : patch.offset[0] + patch.actual_size[0],
                ] = section[
                    patch.offset_padding[1] : patch.offset_padding[1]
                    + patch.actual_size[1],
                    patch.offset_padding[0] : patch.offset_padding[0]
                    + patch.actual_size[0],
                ]

    def mask_only_detected_areas(
        self,
        frames: list[np.ndarray],
        cleaned_frames: list[np.ndarray],
        detections: list[list[DetectionResult]],
    ):
        results: list[np.ndarray] = []
        for frame, cleaned, frame_detections in zip(frames, cleaned_frames, detections):
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for detection in frame_detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

            a = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            b = cv2.bitwise_and(cleaned, cleaned, mask=mask)
            results.append(cv2.add(a, b))
        return results

    # might be a better way to do this since areas that may be clipped by detections are still inpainted
    async def clean(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]] = [],
        detections: list[list[DetectionResult]] = [],
    ) -> list[np.ndarray]:
        ai_cleaned = [x.copy() for x in frames]
        if self.inpaint_patches:
            patches = await asyncio.to_thread(
                self.extract_patches, ai_cleaned, segments
            )
        else:
            patches = [
                _DeepFillV2ImagePatch(
                    frame,
                    frame,
                    mask,
                    np.array([0, 0]),
                    np.array(list(reversed(frame.shape[:2]))),
                    np.array([0, 0]),
                )
                for frame, mask in zip(frames, masks)
            ]
        await asyncio.to_thread(self.clean_patches, patches)
        return await asyncio.to_thread(
            self.mask_only_detected_areas, frames, ai_cleaned, detections
        )

    @staticmethod
    def get_name() -> str:
        return "DeepFillV2"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                "model_path", "Model Path", "Path to the deepfillv2 inpainting model"
            ),
            BooleanPluginArgument(
                "inpaint_patches",
                "InPaint Patches",
                "True to inpaint patches instead of the whole image",
                True,
            ),
            IntPluginArgument(
                "patch_padding",
                "Patch Padding",
                "Padding to apply to patches for inpainting context",
                10,
            ),
            PytorchDevicePluginSelectArgument("device", "Device"),
        ]

    @staticmethod
    def is_valid() -> bool:
        return True
