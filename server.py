import os
import io
import urllib.parse
import requests
import cv2
import numpy as np
import asyncio
from tornado.web import RequestHandler, Application,StaticFileHandler
from threading import Thread
from translator.utils import cv2_to_pil, pil_to_cv2, get_fonts,get_font_path_at_index
from translator.core.pipelines import FullConversion
from translator.core.translators import get_translators
from translator.core.ocr import get_ocr,CleanOcr
from PIL import Image
import json
import re
import webbrowser
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

def cv2_image_from_url(url: str):
    if url.startswith('http'):
        return pil_to_cv2(Image.open(io.BytesIO(requests.get(url).content)))
    else:
        sanitized = urllib.parse.unquote(url.split("?")[0])
        data = cv2.imread(sanitized)
        
        if data is None:
            raise BaseException(f"Failed to load image from path {url}")
        return data

REQUEST_SECTION_REGEX = r"id=([0-9]+)(.*)"
REQUEST_SECTION_PARAMS_REGEX = r"\$([a-z0-9_]+)=([^\/$]+)"

def extract_params(data: str) -> tuple[str,dict]:
    selected_id, params_to_parse = re.findall(REQUEST_SECTION_REGEX,urllib.parse.unquote(data))[0]
    params = {}

    if len(params_to_parse.strip()) > 0:
        for param_name,param_value in re.findall(REQUEST_SECTION_PARAMS_REGEX,params_to_parse.strip()):
            if len(param_value.strip()) > 0:
                params[param_name] = param_value

    return  int(selected_id), params

def send_file_in_chunks(request: RequestHandler,file_path):
    with open(file_path, 'rb') as f:
            while True:
                data = f.read(16384) # or some other nice-sized chunk
                if not data: break
                request.write(data)

class CleanFromWebHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'image/png')

    @run_in_thread
    def get(self):
        try:
            full_url = self.request.full_url()
        
            target_url = full_url[full_url.index("/clean/") + len("/clean/"):]
            
            image_cv2 = cv2_image_from_url(target_url)
            result = clean_image(image_cv2)
            converted_pil = cv2_to_pil(result)
            img_byte_arr = io.BytesIO()
            converted_pil.save(img_byte_arr, format="PNG")
            # Create response given the bytes
            self.write(img_byte_arr.getvalue())
        except:
            self.set_header("Content-Type", 'text/html')
            self.set_status(500)
            traceback.print_exc()
            self.write(traceback.format_exc())
     
class TranslateFromWebHandler(RequestHandler):

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'image/png')

    @run_in_thread
    def get(self,translator_info,ocr_info,font_info):
        try:
            full_url = self.request.full_url()
            
            target_url = "/".join(full_url.split("/")[7:])

            translator_id,translator_params = extract_params(translator_info)
            
            ocr_id,ocr_params = extract_params(ocr_info)

            font_id,_ = extract_params(font_info)

            image_cv2 = cv2_image_from_url(target_url)

            converter = FullConversion(translator=get_translators()[translator_id](**translator_params),ocr=get_ocr()[ocr_id](**ocr_params),font_file=get_font_path_at_index(font_id))

            result = converter([image_cv2])[0]

            converted_pil = cv2_to_pil(result)
            img_byte_arr = io.BytesIO()
            converted_pil.save(img_byte_arr, format="PNG")
            # Create response given the bytes
            self.write(img_byte_arr.getvalue())
        except:
            self.set_header("Content-Type", 'text/html')
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()

class ImageHandler(RequestHandler):

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'image/*')

    @run_in_thread
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
                    send_file_in_chunks(self,item_path)
        except:
            self.set_header("Content-Type", 'text/html')
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()

class BaseHandler(RequestHandler):

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'application/json')

    def get(self):
        try:
            data = { "translators": [], "ocr": [], "fonts": get_fonts()}

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
            self.write(json.dumps(data))
        except:
            self.set_header("Content-Type", 'text/html')
            self.set_status(500)
            self.write(traceback.format_exc())
            traceback.print_exc()

class UiFilesHandler(RequestHandler):

    def initialize(self,build_path) -> None:
        self.build_path = build_path

    @run_in_thread
    def get(self,target_file):
        send_file_in_chunks(self,os.path.join(self.build_path,target_file))

class UiHandler(RequestHandler):
    def get(self):
        self.render("index.html")
        
async def main():
        app_port = 5000
        build_path = os.path.join(os.path.dirname(__file__), "build")
        settings = {
            "template_path": build_path,
            "static_path": os.path.join(build_path,'static'),
        }
        app = Application([
            (r"/",UiHandler),
            (r"/info", BaseHandler),
            (r"/clean/.*",CleanFromWebHandler),
            (r"/translate/(id=[^\/]*)/(id=[^\/]*)/(id=[^\/]*)/.*",TranslateFromWebHandler),
            (r"/images/.*", ImageHandler),
            (r"/(.*)",UiFilesHandler,dict(build_path = build_path)),
            ],**settings)
        app.listen(app_port)
        webbrowser.open(f'http://localhost:{app_port}')
        await asyncio.Event().wait()
        


asyncio.run(main())
