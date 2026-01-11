from manga_translator.core.plugin import Detector, DetectionResult, PluginArgument, StringPluginArgument, PytorchDevicePluginSelectArgument
from ultralytics import YOLO
import torch
import asyncio
from manga_translator.utils import get_default_torch_device,perf_async

class YoloDetector(Detector):
    def __init__(self,model_path: str,device: torch.device = get_default_torch_device()):
        
        super().__init__()
        self.model = YOLO(model=model_path, verbose=False)
        self.device = device

    def predict(self,batch):
        with torch.inference_mode():
            return self.model.predict(batch,device = self.device, verbose=False)
    @perf_async
    async def detect(self, batch):
        with torch.inference_mode():
            results = []
            for prediction in await asyncio.to_thread(self.predict,batch):
                result = []
                boxes = prediction.boxes.xyxy.cpu().int().numpy()
                classes = prediction.boxes.cls.cpu().int()
                confidence = prediction.boxes.conf.cpu()

                for bbox,cls,conf in zip(boxes,classes,confidence):
                    result.append(DetectionResult(cls.item(),bbox,conf.item()))

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