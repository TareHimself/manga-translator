from manga_translator.core.plugin import Cleaner, SegmentationResult,DetectionResult
import numpy as np
import cv2
import asyncio

class AllWhiteCleaner(Cleaner):
    def __init__(self) -> None:
        super().__init__()


    def clean_frame(self,frame: np.ndarray,mask: np.ndarray):
        # Create a white image same size as original
        white = np.full_like(frame, 255, dtype=np.uint8)

        # Replace masked regions with white
        a = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        b = cv2.bitwise_and(white, white, mask=mask)
        c = cv2.add(a,b)
        return c
    
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
        return "Fill White"