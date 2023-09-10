import numpy as np
from numpy import ndarray
from translator.core.plugin import Cleaner
from translator.utils import in_paint_optimized,cv2_to_pil,pil_to_cv2


class LamaCleaner(Cleaner):

    def __init__(self) -> None:
        super().__init__()
        from simple_lama_inpainting import SimpleLama
        self.lama = SimpleLama()

    @staticmethod
    def get_name() -> str:
        return "Lama Cleaner"
    
    def clean(self, frame: ndarray, mask: ndarray, detection_results: list[tuple[tuple[int, int, int, int], str, float]] = ...) -> tuple[ndarray, ndarray]:
        return in_paint_optimized(frame=frame,mask=mask,filtered=detection_results,inpaint_fun= lambda f,m: pil_to_cv2(self.lama(cv2_to_pil(f),cv2_to_pil(m).convert('L')))
                           )
