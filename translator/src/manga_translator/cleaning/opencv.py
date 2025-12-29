from manga_translator.core.plugin import Cleaner, PluginArgument, SegmentationResult,DetectionResult, IntPluginArgument, BooleanPluginArgument
import numpy as np
import cv2
import asyncio

class OpenCvCleaner(Cleaner):
    def __init__(self,radius=1,mask_after_inpaint = False) -> None:
        super().__init__()
        self.radius = radius
        self.mask_after_inpainting = mask_after_inpaint


    def clean_frame(self,frame: np.ndarray,mask: np.ndarray):
        inpaint = cv2.inpaint(frame,mask,inpaintRadius=self.radius,flags=cv2.INPAINT_TELEA)

        if self.mask_after_inpainting:
            # Replace masked regions with white
            a = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            b = cv2.bitwise_and(inpaint, inpaint, mask=mask)
            inpaint = cv2.add(a,b)

        return inpaint
    
    async def clean(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        segments: list[list[SegmentationResult]] = [],
        detections: list[list[DetectionResult]] = []
    ) -> list[np.ndarray]:
        return await asyncio.gather(*[asyncio.to_thread(self.clean_frame,frame,mask) for frame, mask in zip(frames,masks)])
    
    @staticmethod
    def get_name() -> str:
        return "OpenCV"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            IntPluginArgument("radius","radius","",1),
            BooleanPluginArgument("mask_after_inpaint","Mask After Inpaint","",False)
        ]

    @staticmethod
    def is_valid() -> bool:
        return True