import os
import io
import cv2
import asyncio
from manga_translator.utils import pil_to_cv2
from manga_translator.utils import get_default_torch_device
from manga_translator.get import construct_image_to_image_pipeline_from_config
import traceback
import os
import hashlib
from blacksheep import Application, Content, FromFiles, Request, Response, file, get, post, ContentDispositionType, json, bad_request
from blacksheep.client import ClientSession
import numpy as np

print("Using pytorch device",get_default_torch_device())
default_component = {"id": 0, "args": {}}
APP_PORT = 9000
TRANSLATED_IMAGES_PATH = os.path.abspath(os.path.join(".",".temp", "translated"))
CONFIG_PATH = os.path.abspath(os.path.join(".","config.yaml"))
PUBLIC_SERVER_ADDRESS = os.environ.get("PUBLIC_SERVER_ADDRESS", f"http://10.0.0.107:{APP_PORT}")#f"http://127.0.0.1:{APP_PORT}")
pipeline = construct_image_to_image_pipeline_from_config(config_path=CONFIG_PATH)
os.makedirs(TRANSLATED_IMAGES_PATH, exist_ok=True)

PENDING_TRANSLATION_JOBS: dict[str,asyncio.Future] = {}

app = Application()
lock = asyncio.Lock()

app.use_cors(
    allow_methods="*",
    allow_origins="*",
    allow_headers="* Authorization",
    max_age=300,
)

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
    

def compute_hash(data):
    return hashlib.sha256(data).hexdigest()

def bytes_to_mat(data: bytes):
    array = np.asarray(bytearray(data), dtype=np.uint8)
    return cv2.imdecode(array,cv2.IMREAD_COLOR_BGR)

def save_translated(file_path: str,data: np.ndarray):
    cv2.imwrite(file_path,data)

def make_translated_url(key):
    return f"{PUBLIC_SERVER_ADDRESS}/api/v1/translated/{key}.png"

@post("/api/v1/translate")
async def translate_images(files: FromFiles):
    try:
        images = files.value

        if len(images) == 0:
            return bad_request("No Images Sent")
        
        
        keys = await asyncio.gather(*[asyncio.to_thread(
            compute_hash,image.data
        ) for image in images])

        
        file_names = [f"{key}.png" for key in keys]
        file_paths = [os.path.join(TRANSLATED_IMAGES_PATH, file_name) for file_name in file_names]
        files_exist = await asyncio.gather(*[asyncio.to_thread(os.path.exists, file_path) for file_path in file_paths])

        to_translate_indices = [i for i in range(len(files_exist)) if not files_exist[i]]

        loop = asyncio.get_running_loop()

        results: list[asyncio.Future] = [loop.create_future() for _ in keys]

        for i in range(len(files_exist)):
            if files_exist[i]:
                results[i].set_result(make_translated_url(keys[i]))

        if to_translate_indices:
            translation_jobs: list[tuple[int,str,bytes,asyncio.Future]] = []
            async with lock:
                for i in to_translate_indices:
                    key = keys[i]
                    job = PENDING_TRANSLATION_JOBS.get(key,None)
                    if job is None:
                        job = results[i]
                        PENDING_TRANSLATION_JOBS[key] = job
                        translation_jobs.append((i,key,images[i].data,job))
                    else:
                        results[i] = job

            if len(translation_jobs) > 0:
                    try:
                        batch = await asyncio.gather(*[asyncio.to_thread(bytes_to_mat,data) for _,_,data,_ in translation_jobs])

                        translated_batch = await pipeline(batch)

                        await asyncio.gather(*[asyncio.to_thread(save_translated,file_paths[info[0]],result) for info,result in zip(translation_jobs,translated_batch)])
                        async with lock:
                            for _,key,_,pending in translation_jobs:
                                pending.set_result(make_translated_url(key))
                                PENDING_TRANSLATION_JOBS.pop(key)
                    except Exception as e:
                        async with lock:
                            for _,key,_,pending in translation_jobs:
                                pending.set_exception(e)
                                PENDING_TRANSLATION_JOBS.pop(key)
                        raise
            
        urls = await asyncio.gather(*results)
        
        return json({
            "urls": urls
        })
    except Exception as e:
        traceback.print_exc()
        return Response(500,content=Content(b"text/plain",traceback.format_exc().encode()))

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
