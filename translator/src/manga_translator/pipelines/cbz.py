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
from manga_translator.core.plugin import Cleaner, Drawer, OCR, Translator
from manga_translator.drawing.horizontal import HorizontalDrawer
from manga_translator.core.pipeline import Pipeline
from manga_translator.pipelines.image_to_image import (
    ImageToImagePipeline
)


class CbzPipeline(Pipeline):
    def __init__(
        self, image_to_image: ImageToImagePipeline
    ) -> None:
        self.image_to_image = image_to_image

    @staticmethod
    async def read_image(file_path: str) -> np.ndarray:
        return await asyncio.to_thread(cv2.imread,file_path)

    @staticmethod
    async def write_image(file_path: str, image: np.ndarray) -> np.ndarray:
        return await  asyncio.to_thread(cv2.imwrite,file_path, image)
    
    @staticmethod
    def extract_zip(file_path: str, dest_dir: str) -> np.ndarray:
        with zipfile.ZipFile(file_path) as zf:
                # Only keep real files (not directory entries)
                filenames = [zi.filename for zi in zf.infolist() if not zi.is_dir()]
                # Optional: sort for deterministic order (zip order can be arbitrary)
                filenames.sort()
                zf.extractall(dest_dir)
                return filenames
    
    async def __call__(
        self, input_archive: str, output_path: str, batch_size=4
    ) -> bool:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        with tempfile.TemporaryDirectory() as tempdir:
            # Extract everything
            filenames = await asyncio.to_thread(self.extract_zip,input_archive,tempdir)

            out_dir = os.path.abspath(output_path)
            os.makedirs(out_dir, exist_ok=True)

            # Iterate in strides of batch_size and include the tail batch
            for batch_start in range(0, len(filenames), batch_size):
                target_filenames = filenames[batch_start : batch_start + batch_size]
                #batch_no = batch_start // batch_size
                # Read all images in this batch
                images = await asyncio.gather(
                    *[
                        CbzPipeline.read_image(os.path.join(tempdir, name))
                        for name in target_filenames
                    ]
                )

                # Process them
                results = await self.image_to_image(images)

                # Write outputs, creating subdirs if the archive had folders
                write_tasks = []
                for img, rel_name in zip(results, target_filenames):
                    dst_path = os.path.join(out_dir, rel_name)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    write_tasks.append(CbzPipeline.write_image(dst_path, img))

                await asyncio.gather(*write_tasks)

        return True
