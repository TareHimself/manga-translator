from typing import Union
from dotenv import load_dotenv

load_dotenv()
import os
import io
import urllib.parse
import requests
import cv2
import asyncio
from tornado.web import RequestHandler, Application
from manga_translator.utils import cv2_to_pil, pil_to_cv2
from manga_translator.pipelines.image_to_image import ImageToImagePipeline
from manga_translator.get import construct_plugin_by_name
from manga_translator.translation.get import get_translators
from manga_translator.ocr.get import get_ocrs
from manga_translator.drawing.get import get_drawers
from manga_translator.detection.get import get_detectors
from manga_translator.segmentation.get import get_segmenters
from manga_translator.cleaning.get import get_cleaners
from PIL import Image
import json
import re
import webbrowser
import traceback
import os
from operator import itemgetter

default_component = {"id": 0, "args": {}}


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


async def send_file_in_chunks(request: RequestHandler, file_path):
    with open(file_path, "rb") as f:
        while True:
            data = await asyncio.to_thread(
                f.read, 16384
            )  # or some other nice-sized chunk
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

            detector_id, detector_params = itemgetter("id", "args")(
                data.get("detector", default_component)
            )
            segmenter_id, segmenter_params = itemgetter("id", "args")(
                data.get("segmenter", default_component)
            )
            cleaner_id, cleaner_params = itemgetter("id", "args")(
                data.get("cleaner", default_component)
            )

            image_cv2 = pil_to_cv2(Image.open(io.BytesIO(image[0]["body"])))
            converter = ImageToImagePipeline(
                cleaner=construct_plugin_by_name(cleaner_id, cleaner_params),
                detector=construct_plugin_by_name(detector_id, detector_params),
                segmenter=construct_plugin_by_name(segmenter_id, segmenter_params),
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

            translator_id, translator_params = itemgetter("id", "args")(
                data.get("translator", default_component)
            )

            ocr_id, ocr_params = itemgetter("id", "args")(
                data.get("ocr", default_component)
            )

            drawer_id, drawer_params = itemgetter("id", "args")(
                data.get("drawer", default_component)
            )

            detector_id, detector_params = itemgetter("id", "args")(
                data.get("detector", default_component)
            )
            segmenter_id, segmenter_params = itemgetter("id", "args")(
                data.get("segmenter", default_component)
            )
            cleaner_id, cleaner_params = itemgetter("id", "args")(
                data.get("cleaner", default_component)
            )

            image_cv2 = pil_to_cv2(Image.open(io.BytesIO(image[0]["body"])))

            converter = ImageToImagePipeline(
                translator=construct_plugin_by_name(translator_id, translator_params),
                ocr=construct_plugin_by_name(ocr_id, ocr_params),
                drawer=construct_plugin_by_name(drawer_id, drawer_params),
                cleaner=construct_plugin_by_name(cleaner_id, cleaner_params),
                detector=construct_plugin_by_name(detector_id, detector_params),
                segmenter=construct_plugin_by_name(segmenter_id, segmenter_params),
                # color_detect_model=None,
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

    async def get(self):
        try:
            full_url = self.request.full_url()

            item_path = "/".join(full_url.split("/")[4:])

            if item_path.startswith("http"):
                self.write((await asyncio.to_thread(requests.get, item_path)).content)
            else:
                item_path = urllib.parse.unquote(item_path)
                if not os.path.exists(item_path):
                    self.set_status(404)
                else:
                    await send_file_in_chunks(self, item_path)
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

    async def get(self):
        try:
            data = {
                "detectors": [],
                "segmenters": [],
                "translators": [],
                "ocrs": [],
                "drawers": [],
                "cleaners": [],
            }

            detectors = get_detectors()

            for x in range(len(detectors)):
                data["detectors"].append(
                    {
                        "id": detectors[x].__name__,
                        "name": detectors[x].get_name(),
                        "description": detectors[x].__doc__,
                        "args": [x.get() for x in detectors[x].get_arguments()],
                    }
                )

            segmenters = get_segmenters()

            for x in range(len(segmenters)):
                data["segmenters"].append(
                    {
                        "id": segmenters[x].__name__,
                        "name": segmenters[x].get_name(),
                        "description": segmenters[x].__doc__,
                        "args": [x.get() for x in segmenters[x].get_arguments()],
                    }
                )

            translators = get_translators()

            for x in range(len(translators)):
                data["translators"].append(
                    {
                        "id": translators[x].__name__,
                        "name": translators[x].get_name(),
                        "description": translators[x].__doc__,
                        "args": [x.get() for x in translators[x].get_arguments()],
                    }
                )

            ocr = get_ocrs()

            for x in range(len(ocr)):
                data["ocrs"].append(
                    {
                        "id": ocr[x].__name__,
                        "name": ocr[x].get_name(),
                        "description": ocr[x].__doc__,
                        "args": [x.get() for x in ocr[x].get_arguments()],
                    }
                )

            drawers = get_drawers()

            for x in range(len(drawers)):
                data["drawers"].append(
                    {
                        "id": drawers[x].__name__,
                        "name": drawers[x].get_name(),
                        "description": drawers[x].__doc__,
                        "args": [x.get() for x in drawers[x].get_arguments()],
                    }
                )

            cleaners = get_cleaners()

            for x in range(len(cleaners)):
                data["cleaners"].append(
                    {
                        "id": cleaners[x].__name__,
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


class UiHandler(RequestHandler):
    def get(self):
        self.render("index.html")


async def main():
    app_port = 5000
    build_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    settings = {
        "template_path": build_path,
        "static_path": os.path.join(build_path, "assets"),
        "static_url_prefix": "/assets/",
    }
    app = Application(
        [
            (r"/", UiHandler),
            (r"/info", BaseHandler),
            (r"/clean", CleanFromWebHandler),
            (r"/translate", TranslateFromWebHandler),
            # (r"/images/.*", ImageHandler),
            # (r"/mira/translate", MiraTranslateWebHandler),
            # (r"/(.*)", UiFilesHandler, dict(build_path=build_path)),
        ],
        **settings,
    )
    app.listen(app_port)
    webbrowser.open(f"http://localhost:{app_port}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
