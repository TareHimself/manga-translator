from typing import Any
import numpy as np
from PIL import ImageFont, ImageDraw
from numpy import ndarray
from hyphen import Hyphenator
from translator.core.plugin import BasePlugin, PluginArgument,PluginSelectArgument,PluginSelectArgumentOption
from translator.core.translators import TranslatorResult
from translator.utils import get_best_font_size, cv2_to_pil, in_paint_optimized, pil_to_cv2, wrap_text,get_fonts



class Cleaner(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    def clean(self,frame: np.ndarray,mask: np.ndarray,detection_results: list) -> tuple[np.ndarray,np.ndarray]:
        return frame
    
    def __call__(self,frame: np.ndarray,mask: np.ndarray,detection_results: list) -> tuple[np.ndarray,np.ndarray]:
        return self.clean(frame=frame,mask=mask,detection_results=detection_results)
    

class DeepFillV2Cleaner(Cleaner):
    def __init__(self) -> None:
        super().__init__()


    def clean(self, frame: ndarray, mask: ndarray, detection_results: list) -> tuple[ndarray, ndarray]:
        return in_paint_optimized(
            frame,
            mask=mask,
            filtered=detection_results,  # segmentation_results.boxes.xyxy.cpu().numpy()
        )