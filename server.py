from typing import Union
from dotenv import load_dotenv

load_dotenv()
import os
import io
import urllib.parse
import requests
import cv2
import numpy as np
import asyncio
from tornado.web import RequestHandler, Application
from translator.utils import cv2_to_pil, pil_to_cv2, run_in_thread_decorator
from translator.pipelines import FullConversion
from translator.translators.get import get_translators
from translator.translators.deepl import DeepLTranslator
from translator.ocr.get import get_ocr
from translator.ocr.clean import CleanOcr
from translator.ocr.huggingface_ja import JapaneseOcr
from translator.drawers.get import get_drawers
from translator.cleaners.get import get_cleaners
from PIL import Image
import json
import re
import webbrowser
import traceback
import os


def cv2_image_from_url(url: str):
    if url.startswith("http"):
        return pil_to_cv2(Image.open(io.BytesIO(requests.get(url).content)))
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
        for param_name, param_value in re.findall(
            REQUEST_SECTION_PARAMS_REGEX, params_to_parse.strip()
        ):
            if len(param_value.strip()) > 0:
                params[param_name] = param_value

    return int(selected_id), params


def send_file_in_chunks(request: RequestHandler, file_path):
    with open(file_path, "rb") as f:
        while True:
            data = f.read(16384)  # or some other nice-sized chunk
            if not data:
                break
            request.write(data)


class CleanFromWebHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "image/png")

    def options(self):
        self.set_status(200)

    async def post(self):
        try:
            image = self.request.files.get("file")

            if image is None:
                raise BaseException("No Image Sent")

            data = json.loads(self.get_argument("data"))

            cleaner_id, cleaner_params = data.get("cleaner", 0), data.get(
                "cleanerArgs", {}
            )
            image_cv2 = pil_to_cv2(Image.open(io.BytesIO(image[0]["body"])))
            converter = FullConversion(
                ocr=CleanOcr(), cleaner=get_cleaners()[cleaner_id](**cleaner_params)
            )
            results = await converter([image_cv2])
            converted_pil = cv2_to_pil(results[0])
            img_byte_arr = io.BytesIO()
            converted_pil.save(img_byte_arr, format="PNG")
            # Create response given the bytes
            self.write(img_byte_arr.getvalue())
        except:
            self.set_header("Content-Type", "text/html")
            self.set_status(500)
            traceback.print_exc()
            self.write(traceback.format_exc())


class TranslateFromWebHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "image/png")

    def options(self):
        self.set_status(200)

    async def post(self):
        try:
            image = self.request.files.get("file")

            if image is None:
                raise BaseException("No Image Sent")

            data = json.loads(self.get_argument("data"))

            translator_id, translator_params = data.get("translator", 0), data.get(
                "translatorArgs", {}
            )

            ocr_id, ocr_params = data.get("ocr", 0), data.get("ocrArgs", {})

            drawer_id, drawer_params = data.get("drawer", 0), data.get("drawerArgs", {})

            cleaner_id, cleaner_params = data.get("cleaner", 0), data.get(
                "cleanerArgs", {}
            )

            image_cv2 = pil_to_cv2(Image.open(io.BytesIO(image[0]["body"])))

            converter = FullConversion(
                translator=get_translators()[translator_id](**translator_params),
                ocr=get_ocr()[ocr_id](**ocr_params),
                drawer=get_drawers()[drawer_id](**drawer_params),
                cleaner=get_cleaners()[cleaner_id](**cleaner_params),
                color_detect_model=None,
            )

            results = await converter([image_cv2])

            converted_pil = cv2_to_pil(results[0])
            img_byte_arr = io.BytesIO()
            converted_pil.save(img_byte_arr, format="PNG")
            # Create response given the bytes
            self.write(img_byte_arr.getvalue())
        except:
            self.set_header("Content-Type", "text/html")
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()


class ImageHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "image/*")

    @run_in_thread_decorator
    def get(self):
        try:
            full_url = self.request.full_url()

            item_path = "/".join(full_url.split("/")[4:])

            if item_path.startswith("http"):
                self.write(requests.get(item_path).content)
            else:
                item_path = urllib.parse.unquote(item_path)
                if not os.path.exists(item_path):
                    self.set_status(404)
                else:
                    send_file_in_chunks(self, item_path)
        except:
            self.set_header("Content-Type", "text/html")
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()


class BaseHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "application/json")

    def get(self):
        try:
            data = {"translators": [], "ocr": [], "drawers": [], "cleaners": []}

            translators = get_translators()

            for x in range(len(translators)):
                data["translators"].append(
                    {
                        "id": x,
                        "name": translators[x].get_name(),
                        "description": translators[x].__doc__,
                        "args": [x.get() for x in translators[x].get_arguments()],
                    }
                )

            ocr = get_ocr()

            for x in range(len(ocr)):
                data["ocr"].append(
                    {
                        "id": x,
                        "name": ocr[x].get_name(),
                        "description": ocr[x].__doc__,
                        "args": [x.get() for x in ocr[x].get_arguments()],
                    }
                )

            drawers = get_drawers()

            for x in range(len(drawers)):
                data["drawers"].append(
                    {
                        "id": x,
                        "name": drawers[x].get_name(),
                        "description": drawers[x].__doc__,
                        "args": [x.get() for x in drawers[x].get_arguments()],
                    }
                )

            cleaners = get_cleaners()

            for x in range(len(cleaners)):
                data["cleaners"].append(
                    {
                        "id": x,
                        "name": cleaners[x].get_name(),
                        "description": cleaners[x].__doc__,
                        "args": [x.get() for x in cleaners[x].get_arguments()],
                    }
                )
            self.write(json.dumps(data))
        except:
            self.set_header("Content-Type", "text/html")
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()


class MiraTranslateWebHandler(RequestHandler):
    converter: Union[FullConversion, None] = None

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "image/png")

    def options(self):
        self.set_status(200)

    @run_in_thread_decorator
    async def post(self):
        try:
            image = self.request.files.get("file")

            if image is None:
                raise BaseException("No Image Sent")

            if MiraTranslateWebHandler.converter is None:
                MiraTranslateWebHandler.converter = FullConversion(
                    color_detect_model=None,
                    # translator=OpenAiTranslator(api_key=os.getenv("GPT_AUTH")),
                    translator=DeepLTranslator(auth_token=os.getenv("DEEPL_AUTH")),
                    ocr=JapaneseOcr(),
                    translate_free_text=True,
                )

            to_convert = pil_to_cv2(Image.open(io.BytesIO(image[0]["body"])))

            translated = await MiraTranslateWebHandler.converter([to_convert])

            # display_image(translated,"Translated")
            converted_pil = cv2_to_pil(translated[0])

            img_byte_arr = io.BytesIO()

            converted_pil.save(img_byte_arr, format="PNG")
            # Create response given the bytes
            self.set_status(200)

            self.write(img_byte_arr.getvalue())

        except:
            self.set_header("Content-Type", "text/html")
            self.set_status(500)
            traceback.print_exc()
            self.write(traceback.format_exc())


class UiFilesHandler(RequestHandler):
    def initialize(self, build_path) -> None:
        self.build_path = build_path

    @run_in_thread_decorator
    def get(self, target_file):
        send_file_in_chunks(self, os.path.join(self.build_path, target_file))


class UiHandler(RequestHandler):
    def get(self):
        self.render("index.html")


async def main():
    app_port = 5000
    build_path = os.path.join(os.path.dirname(__file__), "ui", "build")
    settings = {
        "template_path": build_path,
        "static_path": os.path.join(build_path, "static"),
    }
    app = Application(
        [
            (r"/", UiHandler),
            (r"/info", BaseHandler),
            (r"/clean", CleanFromWebHandler),
            (r"/translate", TranslateFromWebHandler),
            # (r"/images/.*", ImageHandler),
            (r"/mira/translate", MiraTranslateWebHandler),
            # (r"/(.*)", UiFilesHandler, dict(build_path=build_path)),
        ],
        **settings,
    )
    app.listen(app_port)
    webbrowser.open(f"http://localhost:{app_port}")
    await asyncio.Event().wait()


asyncio.run(main())
