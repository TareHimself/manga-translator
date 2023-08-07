import os
import io
import urllib.parse
import requests
import cv2
import numpy as np
import asyncio
from tornado.web import RequestHandler, Application, StaticFileHandler
from threading import Thread
from translator.utils import cv2_to_pil, pil_to_cv2, get_fonts, get_font_path_at_index
from translator.core.pipelines import FullConversion
from translator.core.translators import HuggingFace
from translator.core.ocr import get_ocr, CleanOcr, MangaOcr
from PIL import Image
import uuid
import re
import traceback


def run_in_thread(func):
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        pending_task = asyncio.Future()

        def run_task():
            nonlocal loop
            loop.call_soon_threadsafe(pending_task.set_result, func(*args, **kwargs))

        Thread(target=run_task, group=None, daemon=True).start()
        result = await pending_task
        return result

    return wrapper


def clean_image(image: np.ndarray):
    converter = FullConversion(ocr=CleanOcr())
    return converter([image])[0]


def cv2_image_from_url(url: str):
    if url.startswith('http'):
        return
    else:
        sanitized = urllib.parse.unquote(url.split("?")[0])
        data = cv2.imread(sanitized)

        if data is None:
            raise BaseException(f"Failed to load image from path {url}")
        return data


REQUEST_SECTION_REGEX = r"id=([0-9]+)(.*)"
REQUEST_SECTION_PARAMS_REGEX = r"\$([a-z0-9_]+)=([^\/$]+)"


def extract_params(data: str) -> tuple[int, dict[str, str]]:
    selected_id, params_to_parse = re.findall(REQUEST_SECTION_REGEX, data)[0]
    params = {}

    if len(params_to_parse.strip()) > 0:
        for param_name, param_value in re.findall(REQUEST_SECTION_PARAMS_REGEX, params_to_parse.strip()):
            if len(param_value.strip()) > 0:
                params[param_name] = param_value

    return int(selected_id), params


def send_file_in_chunks(request: RequestHandler, file_path):
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(16384)  # or some other nice-sized chunk
            if not data:
                break
            request.write(data)


converter = FullConversion(translator=HuggingFace(), ocr=MangaOcr(), font_file=get_font_path_at_index(0))


class TranslateFromWebHandler(RequestHandler):

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'image/png')

    def options(self):
        self.set_status(200)

    @run_in_thread
    def get(self):
        global converter
        try:
            print(f"Received translation request from {self.request.headers.get('X-Forwarded-For')}")

            target_url = self.request.headers.get("X-TRANSLATOR-TARGET")
            print("Headers", self.request.headers)
            print("TARGET URL", target_url)
            request_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/115.0.0.0 Safari/537.36"
            }

            header_start = "X-TRANSLATOR-HEADER-"

            header_keys = list(
                filter(lambda a: a.lower().startswith(header_start.lower()), list(self.request.headers.keys())))
            print(header_keys)
            for header_key in header_keys:
                request_headers[header_key[len(header_start):]] = self.request.headers.get(header_key)

            print("Request headers", request_headers)

            image_cv2 = pil_to_cv2(Image.open(io.BytesIO(requests.get(target_url, headers=request_headers).content)))

            result = converter([image_cv2])[0]

            converted_pil = cv2_to_pil(result)

            img_byte_arr = io.BytesIO()

            converted_pil.save(img_byte_arr, format="PNG")

            self.set_header("Content-Length", len(img_byte_arr.getvalue()))

            # Create response given the bytes
            self.write(img_byte_arr.getvalue())
        except:
            self.set_header("Content-Type", 'text/html')
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()


async def main():
    app_port = 9400
    app = Application([
        (r"/translate", TranslateFromWebHandler),
    ])
    app.listen(app_port)
    await asyncio.Event().wait()


asyncio.run(main())
