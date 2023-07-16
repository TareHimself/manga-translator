import signal
from PIL import Image
import torch
import torchvision.transforms as T
import threading
import os
import queue
import asyncio
import sys

# In this file we queue all inpainting requests so other aspects can run on multiple threads (could also do that here but I am assuming shit gpu)

INPAINT_MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

_inpaint_queue = queue.Queue()


def _inpaint_thread():
    global _inpaint_queue

    current_model = None
    current_model_path = ""

    def get_model(path: str):
        nonlocal current_model_path
        nonlocal current_model

        if path == current_model_path:
            return current_model
        else:
            generator_state_dict = torch.load(path,map_location=INPAINT_MODEL_DEVICE)["G"]

            if "stage1.conv1.conv.weight" in generator_state_dict.keys():
                from .model.networks import Generator
            else:
                from .model.networks_tf import Generator

            # set up network
            generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(
                INPAINT_MODEL_DEVICE
            )

            generator.load_state_dict(generator_state_dict, strict=True)

            current_model_path = path
            current_model = generator
            return generator
        
    def inpaint(
        image: Image,
        mask: Image,
        model_path: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../","../","models", "inpainting.pth"
        ),
    ) -> Image:
        generator = get_model(model_path)

        # prepare input
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask.convert("L"))

        h, w = image.shape[1:]
        grid = 8

        # pad to multiple of grid
        pad_height = grid - h % grid if h % grid > 0 else 0
        pad_width = grid - w % grid if w % grid > 0 else 0

        image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height)).unsqueeze(0)
        mask = torch.nn.functional.pad(mask, (0, pad_width, 0, pad_height)).unsqueeze(0)

        image = (image * 2 - 1.0).to(
            INPAINT_MODEL_DEVICE
        )  # map image values to [-1, 1] range
        mask = (mask > 0.5).to(
            dtype=torch.float32, device=INPAINT_MODEL_DEVICE
        )  # 1.: masked 0.: unmasked

        image_masked = image * (1.0 - mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)  # concatenate channels

        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)

        # complete image
        image_inpainted = image * (1.0 - mask) + x_stage2 * mask

        # convert inpainted image to PIL Image
        img_out = (image_inpainted[0].permute(1, 2, 0) + 1) * 127.5
        img_out = img_out.to(device="cpu", dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())

        # # crop padding
        # img_out = img_out.crop((0, 0, w, h))

        return img_out
    
    payload = _inpaint_queue.get()

    while payload is not None:
        pil_image,pil_mask,model_path,callback = payload
        callback(inpaint(pil_image,pil_mask,model_path))
        payload = _inpaint_queue.get()

pending_tasks = []
async def inpaint_async(image: Image,
        mask: Image,
        model_path: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../","../","models", "inpainting.pth"
        )):
    loop  = asyncio.get_event_loop()

    pending_task = asyncio.Future()

    def callback(inpaint_result):
        nonlocal loop
        loop.call_soon_threadsafe(pending_task.set_result,inpaint_result)

    _inpaint_queue.put((image,mask,model_path,callback))

    pending_tasks.append(pending_task)
    result = await pending_task
    pending_tasks.remove(pending_task)
    return result

def inpaint_threadsafe(image: Image,
        mask: Image,
        model_path: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../","../","models", "inpainting.pth"
        )):
    return asyncio.run(inpaint_async(image,mask,model_path))


_running_thread = threading.Thread(target=_inpaint_thread,group=None)

def _stop_inpaint_thread(signum, frame):
    sys.exit(signum)

_running_thread.start()

signal.signal(signal.SIGINT,_stop_inpaint_thread)
signal.signal(signal.SIGABRT, _stop_inpaint_thread)
signal.signal(signal.SIGTERM, _stop_inpaint_thread)

_og_exit = sys.exit

def _new_sys_exit(*args,**kwargs):
    global _og_exit
    
    try:
        with _inpaint_queue.mutex:
            _inpaint_queue.queue.clear()
        for task in pending_tasks:
            task.cancel()
        _inpaint_queue.put(None)
        _running_thread.join()
    except:
        pass
    _og_exit(*args,**kwargs)
    

sys.exit =  _new_sys_exit
    
