import queue
from typing import Union
import numpy as np
from PIL import Image
from numpy import ndarray
import torch
from translator.core.plugin import Cleaner
from translator.utils import cv2_to_pil, in_paint_optimized, pil_to_cv2, get_model_path
from PIL import Image
from translator.cleaners.deepfillv2_impl import load_model
import torch
import torchvision.transforms as T
import threading
import os
import asyncio
import sys
import atexit


class DeepFillV2Cleaner(Cleaner):
    IN_PAINT_MODEL_DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    )

    DEFAULT_MODEL_PATH = os.path.join("models", "inpainting.pth")

    _job_queue = queue.Queue()

    _model = None

    _model_path = ""

    _pending_tasks = []

    _thread: Union[None, threading.Thread] = None

    @staticmethod
    def get_model(path: str):
        if path == DeepFillV2Cleaner._model_path:
            return DeepFillV2Cleaner._model
        else:
            DeepFillV2Cleaner._model = load_model(
                path, DeepFillV2Cleaner.IN_PAINT_MODEL_DEVICE
            )
            DeepFillV2Cleaner._model_path = path
            return DeepFillV2Cleaner._model

    @staticmethod
    def in_paint(
        image: Image,
        mask: Image,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> Image:
        generator = DeepFillV2Cleaner.get_model(model_path)

        # prepare input
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask.convert("L"))

        h, w = image.shape[1:]
        grid = 8

        # pad to multiple of grid
        pad_height = grid - h % grid if h % grid > 0 else 0
        pad_width = grid - w % grid if w % grid > 0 else 0

        image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height)).unsqueeze(
            0
        )
        mask = torch.nn.functional.pad(mask, (0, pad_width, 0, pad_height)).unsqueeze(0)

        image = (image * 2 - 1.0).to(
            DeepFillV2Cleaner.IN_PAINT_MODEL_DEVICE
        )  # map image values to [-1, 1] range
        mask = (mask > 0.5).to(
            dtype=torch.float32, device=DeepFillV2Cleaner.IN_PAINT_MODEL_DEVICE
        )  # 1.: masked 0.: unmasked

        image_masked = image * (1.0 - mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat(
            [image_masked, ones_x, ones_x * mask], dim=1
        )  # concatenate channels

        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)

        # complete image
        image_in_painted = image * (1.0 - mask) + x_stage2 * mask

        # convert in_painted image to PIL Image
        img_out = (image_in_painted[0].permute(1, 2, 0) + 1) * 127.5
        img_out = img_out.to(device="cpu", dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())

        # # crop padding
        # img_out = img_out.crop((0, 0, w, h))

        return img_out

    @staticmethod
    def _in_paint_thread():
        payload = DeepFillV2Cleaner._job_queue.get()

        while payload is not None:
            data = payload
            pil_image, pil_mask, model_path, callback = data

            callback(DeepFillV2Cleaner.in_paint(pil_image, pil_mask, model_path))

            payload = DeepFillV2Cleaner._job_queue.get()

    @staticmethod
    async def add_in_paint_task(
        image: np.ndarray, mask: np.ndarray, model_path: str = DEFAULT_MODEL_PATH
    ):
        loop = asyncio.get_event_loop()

        pending_task = asyncio.Future()

        def callback(in_paint_result):
            nonlocal loop

            loop.call_soon_threadsafe(pending_task.set_result, pil_to_cv2(in_paint_result))

        DeepFillV2Cleaner._job_queue.put((cv2_to_pil(image), cv2_to_pil(mask), model_path, callback))
        DeepFillV2Cleaner._pending_tasks.append(pending_task)

        result = await pending_task
        DeepFillV2Cleaner._pending_tasks.remove(pending_task)
        return result

    @staticmethod
    def _stop_in_paint_thread(signum, frame):
        sys.exit(signum)

    @staticmethod
    def _exit_thread():
        try:
            for task in DeepFillV2Cleaner._pending_tasks:
                task.cancel()
            # callback_queue.put(None)
            if DeepFillV2Cleaner._thread.is_alive():
                DeepFillV2Cleaner._job_queue.put(None)
                DeepFillV2Cleaner._thread.join()
        except:
            pass

    def __init__(self) -> None:
        super().__init__()

        if DeepFillV2Cleaner._thread is None:
            DeepFillV2Cleaner._thread = threading.Thread(
                target=lambda: DeepFillV2Cleaner._in_paint_thread(),
                group=None,
                daemon=True,
            )
            DeepFillV2Cleaner._thread.start()
            atexit.register(DeepFillV2Cleaner._exit_thread)

    @staticmethod
    def get_name() -> str:
        return "Deep Fill V2"

    async def clean(
        self,
        frame: ndarray,
        mask: ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = [],
    ) -> tuple[ndarray, ndarray]:
        loop = asyncio.get_event_loop()
        return await in_paint_optimized(
            frame,
            mask=mask,
            filtered=detection_results,  # segmentation_results.boxes.xyxy.cpu().numpy()
            inpaint_fun=lambda frame, mask: loop.create_task(DeepFillV2Cleaner.add_in_paint_task(
                    frame, mask
                )),
        )
