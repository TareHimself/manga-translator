import numpy as np

from comic_localizer.core.constants import SegmentationType
from comic_localizer.core.plugin import (
    DetectionResult,
    PluginArgument,
    PytorchDevicePluginArgument,
    StringPluginArgument,
    Segmenter,
    SegmentationResult,
)
from ultralytics import YOLO
import torch
import asyncio
from comic_localizer.utils import get_default_torch_device


class YoloSegmenter(Segmenter):
    def __init__(
        self, model_path: str, device: torch.device = get_default_torch_device()
    ):

        super().__init__()
        self.model = YOLO(model=model_path, verbose=False)
        self.device = device

    @staticmethod
    def conv_cls(cls_id: int):
        try:
            return SegmentationType(cls_id)
        except ValueError:
            return SegmentationType.Text

    def predict(self, batch):
        with torch.inference_mode():
            results = []
            for prediction in self.model.predict(
                [
                    x[..., ::-1] for x in batch
                ],  # flip RGB to BGR since ultralytics expects BGR
                conf=0.1,
                device=self.device,
                verbose=False,
            ):
                result = []

                if prediction.masks is not None and prediction.boxes is not None:
                    classes = prediction.boxes.cls.cpu().int()  # type: ignore[union-attr]
                    confidence = prediction.boxes.conf.cpu()  # type: ignore[union-attr]
                    masks = prediction.masks.xy

                    for mask, cls, conf in zip(masks, classes, confidence):
                        result.append(
                            SegmentationResult(
                                YoloSegmenter.conv_cls(int(cls.item())),
                                mask.astype(int),
                                conf.item(),
                            )
                        )

                results.append(result)

            return results

    async def segment(
        self,
        frames: list[np.ndarray],
        detections: list[list[DetectionResult]],
    ) -> list[list[SegmentationResult]]:
        return await asyncio.to_thread(self.predict, frames)

    @staticmethod
    def get_name() -> str:
        return "Yolo"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument("model_path", "Model Path", "Path to the yolo model"),
            PytorchDevicePluginArgument("device", "Device"),
        ]
