import numpy as np
import torch
from translator.color_detect.constants import IMAGE_SIZE
from translator.color_detect.models import get_color_detection_model
from translator.color_detect.utils import generate_color_detection_train_example, apply_transforms
from translator.utils import display_image
import cv2
import os
from faker import Faker
from random import Random
from translator.color_detect.train import train_model

pytorch_device = torch.device("cuda:0")

backgrounds = [cv2.imread(f"./assets/backgrounds/{x}") for x in
               os.listdir("./assets/backgrounds")]  # some background noise for the dataset


model = train_model(epochs=30000, seed=20, device=pytorch_device, num_samples=50000,#30000,
                    num_workers=1,batch_size=64,backgrounds=backgrounds,patience=100)#, weights_path="models/color_detection.pt")  # trains then returns the trained model

model = model.to(torch.device('cpu'))

torch.save(model.state_dict(), "trained.pt")

#model = get_color_detection_model(weights_path="models/color_detection.pt")

model = model.to(pytorch_device)

model.eval()

bg_random = (np.random.rand(500,500,3) * 255).astype(np.uint8)
# with torch.no_grad():
#     with torch.inference_mode():
#         test_img = cv2.imread(f"./Screenshot 2024-01-16 120841.png")
#         to_eval = apply_transforms(test_img).unsqueeze(0).type(torch.FloatTensor).to(pytorch_device)
#         results = model(to_eval)[0].cpu().numpy()
#         results[:-1] = results[:-1] * 255
#         color = np.array(results, dtype=np.int32)
#         print("Detected color", color)
#         display_image(test_img, "Test Frame")
with torch.no_grad():
    with torch.inference_mode():
        seed = 90
        fake_en = Faker(['ja_JP'])
        fake_en.seed_instance(seed)
        gen = Random(seed)
        while True:
            try:
                text = " ".join([fake_en.name() for x in range(gen.randint(1, 4))])
                example, example_color = generate_color_detection_train_example(text, size=(gen.randrange(round(IMAGE_SIZE / 2),IMAGE_SIZE * 2), gen.randrange(round(IMAGE_SIZE / 2),IMAGE_SIZE * 2)),
                                                                        # background=gen.choice(backgrounds),
                                                                        generator=gen, font_file="fonts/NotoSansJP-Regular.ttf",background=bg_random)
                to_eval = example.copy()
                to_eval = apply_transforms(to_eval).unsqueeze(0).type(torch.FloatTensor).to(pytorch_device)
                results = model(to_eval)[0].cpu().numpy()
                results[:-1] = results[:-1] * 255
                #results = results * 255
                #color = np.array(results, dtype=np.int32)
                print("Detected color", results[:-1].astype(np.uint8),results[-1])
                print("Actual color", example_color)
                display_image(example, "Test Frame")
            except KeyboardInterrupt:
                break
