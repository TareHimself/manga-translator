import numpy as np
from manga_translator.core.plugin import PluginArgument, PytorchDevicePluginSelectArgument, StringPluginArgument, Segmenter, SegmentationResult,Point
from ultralytics import YOLO
import torch
import asyncio
from manga_translator.utils import get_default_torch_device,perf_async


class YoloSegmenter(Segmenter):
    def __init__(self,model_path: str,device: torch.device = get_default_torch_device()):
        
        super().__init__()
        self.model = YOLO(model=model_path, verbose=False)
        self.device = device

    def predict(self,batch):
        with torch.inference_mode():
            return self.model.predict(batch,device = self.device, verbose=False)
    
    @perf_async
    async def segment(self, batch):
        with torch.inference_mode():
            results = []
            for prediction in await asyncio.to_thread(self.predict,batch):
                result = []
                
                if prediction.masks is not None:
                    classes = prediction.boxes.cls.cpu().int()
                    confidence = prediction.boxes.conf.cpu()
                    masks = prediction.masks.xy

                    for mask,cls,conf in zip(masks,classes,confidence):
                        result.append(SegmentationResult(cls.item(),mask.astype(int),conf.item()))

                results.append(result)

            return results
        
    @staticmethod
    def get_name() -> str:
        return "Yolo"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument("model_path","Model Path","Path to the yolo model"),
            PytorchDevicePluginSelectArgument("device","Device")
        ]