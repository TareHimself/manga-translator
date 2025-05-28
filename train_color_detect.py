import torch
import random
import numpy as np
from faker import Faker
from translator.color_detect.training.train import train_model
from translator.color_detect.model import get_color_detection_model
from translator.color_detect.training.utils import (
    generate_color_detection_train_example,
    format_color_detect_output,
    apply_transforms,
    lab_to_rgb_np,
    rgb_to_lab_np,
    normalize_lab_np,
    denormalize_lab_np,
)
from translator.color_detect.training.dataset import ColorDetectionDataset

device = torch.device("cuda:0")

# dataset = ColorDetectionDataset(generate_target=20000,cache_dir="./color_data")
# model = train_model(dataset= dataset, device=device)

# model = model.to(torch.device("cpu"))

# torch.save(model.state_dict(), "trained.pt")

model = get_color_detection_model(weights_path="trained.pt", device=device)


model = model.to(device)

model.eval()


bg_random = (np.random.rand(500, 500, 3) * 255).astype(np.uint8)
with torch.no_grad():
    with torch.inference_mode():
        seed = 90
        fake_en = Faker(["ja_JP"])
        fake_en.seed_instance(seed)
        gen = random.Random(seed)
        while True:
            try:
                text = " ".join([fake_en.name() for x in range(gen.randint(1, 4))])
                example, example_color = generate_color_detection_train_example(
                    text,
                    size=(128, 128),
                    generator=gen,
                    font_file="fonts/NotoSansJP-Regular.ttf",
                    background=bg_random,
                )
                to_eval = example.copy()
                to_eval = (
                    apply_transforms(to_eval)
                    .unsqueeze(0)
                    .type(torch.FloatTensor)
                    .to(device)
                )
                r = format_color_detect_output(model(to_eval))
                # results = results * 255
                # color = np.array(results, dtype=np.int32)
                print("Detected color", r[0])
                print("Actual color", example_color)
                input("Press any key to continue")
            except KeyboardInterrupt:
                break
