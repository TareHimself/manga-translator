from PIL import Image
import torch
import torchvision.transforms as T
import os

INPAINT_MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_model = None
current_model_path = ""


def get_model(path: str):
    global current_model_path
    global current_model
    if path == current_model_path:
        return current_model
    else:
        generator_state_dict = torch.load(path)["G"]

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
