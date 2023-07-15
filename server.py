import io
import requests
import cv2
import numpy as np
import asyncio
from tornado.web import RequestHandler, Application
from threading import Thread
from translator.utils import cv2_to_pil, pil_to_cv2
from translator.pipelines import FullConversion
from translator.translators import get_translators
from translator.ocr import get_ocr,CleanOcr
from PIL import Image
import json
import re
import traceback

def run_in_thread(func):
    async def wrapper(*args,**kwargs):
        loop  = asyncio.get_event_loop()
        pending_task = asyncio.Future()

        def run_in_thread():
            nonlocal loop
            loop.call_soon_threadsafe(pending_task.set_result,func(*args,**kwargs))

        Thread(target=run_in_thread,group=None,daemon=True).start()
        result = await pending_task
        return result
    return wrapper

def clean_image(image: np.ndarray):
    converter = FullConversion(ocr=CleanOcr())
    return converter([image])[0]

# @app.route("/clean", methods=["POST"])
# def clean():
#     imagefile = flask.request.files.get("image")
#     image_cv2 = pil_to_cv2(Image.open(io.BytesIO(imagefile.read())))
#     converted = clean_image(image_cv2)
#     converted_pil = cv2_to_pil(converted)
#     img_byte_arr = io.BytesIO()
#     converted_pil.save(img_byte_arr, format="PNG")
#     # Create response given the bytes
#     response = flask.make_response(img_byte_arr.getvalue())
#     response.headers.set("Content-Type", "image/png")
#     return response

# @app.route("/clean/<path:path>'", methods=["POST"])
# def clean_from_url(path):
#     print(path)
#     return "Hi"

# @app.route("/", methods=["GET"])
# def index():
    

#     response = flask.make_response(json.dumps(data))
#     response.headers.set("Content-Type", "application/json")
#     return response

class CleanFromWebHandler(RequestHandler):
    @run_in_thread
    def get(self):
        full_url = self.request.full_url()
        
        target_url = full_url[full_url.index("/clean/") + len("/clean/"):]
        
        image_cv2 = pil_to_cv2(Image.open(io.BytesIO(requests.get(target_url).content)))
        result = clean_image(image_cv2)
        converted_pil = cv2_to_pil(result)
        img_byte_arr = io.BytesIO()
        converted_pil.save(img_byte_arr, format="PNG")
        # Create response given the bytes
        self.set_header("Content-Type", "image/png")
        self.write(img_byte_arr.getvalue())

REQUEST_SECTION_REGEX = r"id=([0-9]+)(.*)"
REQUEST_SECTION_PARAMS_REGEX = r"\$([a-z0-9_]+)=([^\/$]+)"

def extract_params(data: str) -> tuple[str,dict]:
    selected_id, params_to_parse = re.findall(REQUEST_SECTION_REGEX,data)[0]
    params = {}

    if len(params_to_parse.strip()) > 0:
        for param_name,param_value in re.findall(REQUEST_SECTION_PARAMS_REGEX,params_to_parse.strip()):
            params[param_name] = param_value

    return  int(selected_id), params
     
class TranslateFromWebHandler(RequestHandler):
    @run_in_thread
    def get(self,translator_info,ocr_info):
        try:
            full_url = self.request.full_url()
            
            target_url = "/".join(full_url.split("/")[6:])

            translator_id,translator_params = extract_params(translator_info)

            ocr_id,ocr_params = extract_params(ocr_info)


            converter = FullConversion(translator=get_translators()[translator_id](**translator_params),ocr=get_ocr()[ocr_id](**ocr_params))

            image_cv2 = pil_to_cv2(Image.open(io.BytesIO(requests.get(target_url).content)))
            result = converter([image_cv2])[0]
            converted_pil = cv2_to_pil(result)
            img_byte_arr = io.BytesIO()
            converted_pil.save(img_byte_arr, format="PNG")
            # Create response given the bytes
            self.set_header("Content-Type", "image/png")
            self.write(img_byte_arr.getvalue())
        except Exception as e:
            traceback.print_exc()
            self.set_status(500)
            self.finish()
          
class BaseHandler(RequestHandler):
    def get(self):
        try:
            data = { "translators": [], "ocr": []}

            translators = get_translators()

            for x in range(len(translators)):
                data["translators"].append({
                "id": x,
                "name": translators[x].get_name(),
                "description": translators[x].__doc__,
                "args": [x.get() for x in translators[x].get_arguments()]
                })

            ocr = get_ocr()

            for x in range(len(ocr)):
                data["ocr"].append({
                "id": x,
                "name": ocr[x].get_name(),
                "description": ocr[x].__doc__,
                "args": [x.get() for x in ocr[x].get_arguments()]
                })

            self.set_header("Content-Type", 'application/json')
            self.write(json.dumps(data))
        except Exception as e:
            traceback.print_exc()
            self.set_status(500)
            self.finish()

async def main():
        app = Application([
            (r"/info", BaseHandler),
            (r"/clean/.*",CleanFromWebHandler),
            (r"/translate/(id=[^\/]*)/(id=[^\/]*)/.*",TranslateFromWebHandler)
            ])
        app.listen(5000)
        await asyncio.Event().wait()


asyncio.run(main())
