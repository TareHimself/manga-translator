import os
import io
import cv2
import asyncio
from manga_translator.utils import pil_to_cv2
from manga_translator.utils import get_default_torch_device
from manga_translator.get import construct_image_to_image_pipeline_from_config
from PIL import Image
import os
import hashlib
from blacksheep import Application, Content, FromFiles, Request, Response, file, get, post, ContentDispositionType
from blacksheep.client import ClientSession
import numpy as np


print("Using pytorch device",get_default_torch_device())
default_component = {"id": 0, "args": {}}
APP_PORT = 9000
TRANSLATED_IMAGES_PATH = os.path.abspath(os.path.join(".",".temp", "translated"))
CONFIG_PATH = os.path.abspath(os.path.join(".","config.yaml"))
PUBLIC_SERVER_ADDRESS = os.environ.get("PUBLIC_SERVER_ADDRESS", f"http://127.0.0.1:{APP_PORT}")
pipeline = construct_image_to_image_pipeline_from_config(config_path=CONFIG_PATH)
os.makedirs(TRANSLATED_IMAGES_PATH, exist_ok=True)

PENDING_TRANSLATION_JOBS: dict[str,asyncio.Future] = {}

app = Application()
lock = asyncio.Lock()

# I cant figure out how to send requests these requests from the extension
@post("/api/v1/get-image")
async def fetch_image(request: Request):
    data = await request.json()
    async with ClientSession() as client:
        proxy_response = await client.get(data["url"],headers=data["headers"])

        assert proxy_response is not None
        
        resp = Response(proxy_response.status,[],Content(proxy_response.content_type(),await proxy_response.read()))

        for header in proxy_response.headers.keys():
            if header.lower() in (b"content-length", b"transfer-encoding"):
                continue
            resp.headers.add(header,proxy_response.headers.get_single(header))
        
        return resp
    

@post("/api/v1/translate")
async def translate_images(files: FromFiles):
    images = files.value

    if len(images) == 0:
        raise BaseException("No Image Sent")
    
    image = images[0]

    byte_hash = await asyncio.to_thread(
        hashlib.sha256,image.data
    )

    key = byte_hash.hexdigest()
    dest_filename = f"{key}.png"
    dest_file = os.path.join(TRANSLATED_IMAGES_PATH, dest_filename)
    
    if not await asyncio.to_thread(os.path.exists, dest_file):
        pending = None
        is_leader = False
        async with lock:
            pending = PENDING_TRANSLATION_JOBS.get(key,None)
            if pending is None:
                is_leader = True
                pending = asyncio.get_running_loop().create_future()
                PENDING_TRANSLATION_JOBS[key] = pending

        if is_leader:
                try:
                    array = np.asarray(bytearray(image.data), dtype=np.uint8)
                    image_cv2 = cv2.imdecode(array,cv2.IMREAD_COLOR_BGR)
                    results = await pipeline(
                        [image_cv2]
                    )  # TODO: Maybe add some kind of batching here

                    await asyncio.to_thread(cv2.imwrite, dest_file, results[0])
                    PENDING_TRANSLATION_JOBS[key].set_result()
                except Exception as e:
                    PENDING_TRANSLATION_JOBS[key].set_exception(e)
                finally:
                    PENDING_TRANSLATION_JOBS.pop(key)
        else:
            await pending
    
    return Response(200,None,Content(b"text/plain",f"{PUBLIC_SERVER_ADDRESS}/api/v1/translated/{dest_filename}".encode()))

@get("/api/v1/translated/{key}")
async def get_translated_image(key: str):
    file_path = os.path.join(TRANSLATED_IMAGES_PATH, key)

    if not os.path.isabs(file_path):
        return Response(400)
    
    if await asyncio.to_thread(os.path.exists, file_path):
        return file(value=file_path,content_type="image/png",file_name= key, content_disposition=ContentDispositionType.INLINE)
    else:
        Response(404)

@app.on_start
async def on_startup(app: Application):
    print(f"Running at {PUBLIC_SERVER_ADDRESS}/api/v1")
