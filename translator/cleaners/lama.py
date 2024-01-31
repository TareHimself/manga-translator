import numpy as np
from numpy import ndarray
import asyncio
from translator.core.plugin import Cleaner, PluginArgument, PluginTextArgument
from translator.utils import in_paint_optimized, cv2_to_pil, pil_to_cv2


class LamaCleaner(Cleaner):
    def __init__(self, dilation="9") -> None:
        super().__init__()
        from simple_lama_inpainting import SimpleLama

        self.lama = SimpleLama()
        self.dilation = int(dilation)

    @staticmethod
    def get_name() -> str:
        return "Lama Cleaner"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(id="dilation", name="Mask Dilation",description="The dilation used for the text mask", default="9")]
    
    def clean_with_lama(self,frame,mask):
        return pil_to_cv2(
                self.lama(cv2_to_pil(frame), cv2_to_pil(mask).convert("L"))
            )
    
    async def clean(
        self,
        frame: ndarray,
        mask: ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = ...,
    ) -> tuple[ndarray, ndarray]:
        return in_paint_optimized(
            frame=frame,
            mask=mask,
            filtered=detection_results,
            mask_dilation_kernel_size=self.dilation,
            inpaint_fun=lambda f, m: self.clean_with_lama(f,m),
        )
