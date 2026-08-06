# Adapted from https://github.com/enesmsahin/simple-lama-inpainting , maybe setup our own repo in the future since I can't make sense of the original lama repo
from comic_localizer.core.plugin import (
    Cleaner,
    PluginArgument,
    PytorchDevicePluginArgument,
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
from comic_localizer.utils import get_default_torch_device
from collections import defaultdict


class _AiImagePatch:
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


class PatchedAiCleaner(Cleaner):
    """
    Base class for ai based cleaners that support patch inpaining.
    """

    def __init__(
        self,
        model_path: str,
        inpaint_patches=True,
        patch_padding=4,
        device: torch.device = get_default_torch_device(),
        grouping_bucket_size: int = 512,
        max_group_pixels: int = 2048 * 2048 * 4,
    ) -> None:
        super().__init__()
        self.device = device
        # For some reason this is very slow
        # self.model = torch.jit.freeze(
        #     torch.jit.load(model_path, map_location=device).eval()
        # )
        self.model = self.load_model(model_path, self.device)
        self.inpaint_patches = inpaint_patches
        self.patch_padding = np.array([patch_padding, patch_padding], dtype=np.int32)
        self.grouping_bucket_size = grouping_bucket_size
        self.max_group_pixels = max_group_pixels
        self.zero_max = np.zeros((2), dtype=np.int32)

    def load_model(self, model_path: str, device: torch.device) -> torch.jit.ScriptModule:
        return torch.jit.load(model_path, map_location=device).eval()  # type: ignore[return-value]

    def group_patches(self, patches: list[_AiImagePatch]):
        # we can do some kind of size based grouping to batch here
        groups = defaultdict(list)

        for patch in patches:
            h, w, _ = patch.patch.shape

            key = (h // self.grouping_bucket_size, w // self.grouping_bucket_size)

            groups[key].append(patch)

        final_groups: list[list[_AiImagePatch]] = []

        for key in groups:
            a: list[_AiImagePatch] = []
            max_h = 0
            max_w = 0
            for patch in groups[key]:
                h, w, _ = patch.patch.shape
                new_max_h = max(max_h, h)
                new_max_w = max(max_w, w)
                # every patch in the group gets padded up to the group's max
                # dimensions before being stacked into one batch tensor, so
                # the real cost of adding this patch is (count so far + 1)
                # patches all at the new max size -- not this patch's own
                # (unpadded) size
                projected_pixels = (
                    (len(a) + 1) * new_max_h * new_max_w * 4
                )  # 3 channels for image, 1 for mask
                if projected_pixels > self.max_group_pixels and len(a) > 0:
                    final_groups.append(a)
                    a = [patch]
                    max_h, max_w = h, w
                else:
                    a.append(patch)
                    max_h, max_w = new_max_h, new_max_w

            if len(a) > 0:
                final_groups.append(a)
        return final_groups

    @staticmethod
    def to_factor(number: int, factor: int):
        remainder = number % factor
        return number + (0 if remainder == 0 else factor - remainder)

    def process_input_batched(self, batch: list[np.ndarray]) -> torch.Tensor:
        tensors = [torch.from_numpy(x).permute(2, 0, 1) for x in batch]
        max_shape = np.stack([np.array(t.shape[1:]) for t in tensors]).max(axis=0)
        h, w = max_shape
        h = self.to_factor(h, 8)
        w = self.to_factor(w, 8)

        stacked_tensors = torch.stack(
            [
                F.pad(
                    tensor,
                    (0, w - tensor.shape[2], 0, h - tensor.shape[1]),
                    mode="replicate",
                )
                for tensor in tensors
            ],
            dim=0,
        )

        return stacked_tensors / 255.0

    def process_masks_batched(self, batch: list[np.ndarray]) -> torch.Tensor:
        tensors = [torch.from_numpy(x).unsqueeze(0) for x in batch]
        max_shape = np.stack([np.array(t.shape) for t in tensors]).max(axis=0)
        _, h, w = max_shape
        h = self.to_factor(h, 8)
        w = self.to_factor(w, 8)

        stacked_tensors = torch.stack(
            [
                F.pad(
                    tensor,
                    (0, w - tensor.shape[2], 0, h - tensor.shape[1]),
                    mode="constant",
                    value=0,
                )
                for tensor in tensors
            ],
            dim=0,
        )

        return stacked_tensors / 255.0

    def process_output_batched(self, x: torch.Tensor) -> np.ndarray:
        result = (
            (x * 255).clip(0, 255).byte().permute(0, 2, 3, 1)
        ).numpy()
        return result

    def extract_patches(
        self, frames: list[np.ndarray], segments: list[list[SegmentationResult]]
    ) -> list[_AiImagePatch]:
        patches: list[_AiImagePatch] = []

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
                mask_patch = np.zeros(
                    (mask_patch_size[1], mask_patch_size[0]), dtype=frame.dtype
                )
                cv2.fillPoly(
                    mask_patch,
                    [np.array([point - p1 for point in segment.points])],
                    (255, 255, 255),
                )

                patches.append(
                    _AiImagePatch(
                        frame,
                        frame_patch,
                        mask_patch,
                        actual_p1,
                        actual_size,
                        final_offfset_padding,
                    )
                )

        return patches

    def do_inference(self, patches: list[list[_AiImagePatch]]):
        with torch.inference_mode():
            # we can do some kind of size based grouping to batch here
            inputs = []
            outputs = []
            for patch_group in patches:
                inputs.append(
                    (
                        self.process_input_batched([x.patch for x in patch_group]),
                        self.process_masks_batched([x.mask for x in patch_group]),
                    )
                )

            # with get_autocast(self.device,self.fp16):
            for batch, masks in inputs:
                masks = masks.to(self.device)
                batch = batch.to(self.device)
                outputs.append(self.model(batch, masks).cpu())

            outputs = [self.process_output_batched(x) for x in outputs]

            for patch_group, output_batch in zip(patches, outputs):
                for patch, out_image in zip(patch_group, output_batch):
                    h, w = patch.patch.shape[:2]
                    section = out_image[0:h, 0:w]
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
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255,), thickness=-1)

            a = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            b = cv2.bitwise_and(cleaned, cleaned, mask=mask)
            results.append(cv2.add(a, b))
        return results

    # might be a better way to do this since areas that may be clipped by detections are still inpainted
    def clean_sync(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]],
        detections: list[list[DetectionResult]],
    ) -> list[np.ndarray]:

        ai_cleaned = [x.copy() for x in frames]
        if self.inpaint_patches:
            patches = self.extract_patches(ai_cleaned, segments)
        else:
            patches = [
                _AiImagePatch(
                    frame,
                    frame,
                    mask,
                    np.array([0, 0]),
                    np.array(list(reversed(frame.shape[:2]))),
                    np.array([0, 0]),
                )
                for frame, mask in zip(ai_cleaned, masks)
            ]

        grouped_patches = self.group_patches(patches)
        self.do_inference(grouped_patches)

        return self.mask_only_detected_areas(frames, ai_cleaned, detections)

    async def clean(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]],
        detections: list[list[DetectionResult]],
    ) -> list[np.ndarray]:
        return await asyncio.to_thread(
            self.clean_sync, frames, masks, segments, detections
        )

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                "model_path", "Model Path", "Path to the torch script model"
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
                4,
            ),
            PytorchDevicePluginArgument("device", "Device"),
            IntPluginArgument(
                "grouping_bucket_size",
                "Grouping Bucket Size",
                "Pixel bucket size for grouping",
                512,
            ),
            IntPluginArgument(
                "max_group_pixels",
                "Max Pixels Per Group",
                "Maximum number of pixels in a group (i.e. sum([width * height for image in group]))",
                2048 * 2048 * 4,
            ),
        ]

    @staticmethod
    def is_valid() -> bool:
        return True
