import torch
import os
import math
import zipfile
import tempfile
import numpy as np
import asyncio
import cv2
import traceback
from typing import Union
from translator.cleaners.deepfillv2 import DeepFillV2Cleaner
from translator.core.plugin import Cleaner, Drawer, Ocr, Translator
from translator.drawers.horizontal import HorizontalDrawer
from translator.pipelines.pipeline import Pipeline
from translator.pipelines.image_to_image import ImageToImagePipeline,DefaultImageToImagePipeline
from translator.utils import get_model_path,run_in_thread

class CbzPipeline(Pipeline):
    def __init__(self,image_to_image : ImageToImagePipeline = DefaultImageToImagePipeline) -> None:
        self.image_to_image = image_to_image


    
    @staticmethod
    async def read_image(file_path: str) -> np.ndarray:
        return await run_in_thread(lambda : cv2.imread(file_path))
    
    @staticmethod
    async def write_image(file_path: str,image: np.ndarray) -> np.ndarray:
        print("WRITING FILE TO",file_path)
        return await run_in_thread(lambda : cv2.imwrite(file_path,image))

    async def __call__(self,input_archive: str,output_path: str,batch_size=4) -> bool:
        with tempfile.TemporaryDirectory() as tempdir:
            filenames = []
            with zipfile.ZipFile(input_archive) as f:
                filenames = f.namelist()
                f.extractall(tempdir)

            idx = 0

            out_dir = os.path.abspath(output_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)


            while idx + batch_size < len(filenames) - 1:
                target_filenames = filenames[idx:idx + batch_size]
                idx += batch_size
                try:
                    images = await asyncio.gather(*[CbzPipeline.read_image(os.path.join(tempdir,x)) for x in target_filenames])
                    results = await self.image_to_image(images)
                    await asyncio.gather(*[CbzPipeline.write_image(os.path.join(out_dir,filename),img) for img,filename in zip(results,target_filenames)])
                except:
                    print(f"Failed batch {math.floor((idx - batch_size) / batch_size)}")
                    traceback.print_exc()
        return True


        
            


        