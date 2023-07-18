import numpy as np
import torch
from translator.utils import generate_color_detection_train_example,display_image
import cv2
import os
from faker import Faker
from random import Random
from translator.color_detect.train import train_model
from translator.color_detect.models import get_color_detection_model
from translator.color_detect.datasets import ColorDetectionDataset
from translator.utils import transform_sample

pytorch_device = torch.device("cuda:0")

backgrounds = [cv2.imread(f"./backgrounds/{x}") for x in os.listdir("./backgrounds")] # some background noise for the dataset

model = train_model(epochs=10000,backgrounds=backgrounds,seed=30,train_device=pytorch_device,num_samples=6000,num_workers=5) # trains then returns the trained model

torch.save(model.state_dict(),"trained.pt")

# model = get_color_detection_model(weights_path="trained_good.pt")

# model = model.to(pytorch_device)

model.eval()

with torch.no_grad():
    with torch.inference_mode():
        seed = 10
        fake_en = Faker(['ja_JP'])
        fake_en.seed_instance(seed)
        gen = Random(seed)
        while True:
            try:
                text = " ".join([fake_en.name() for x in range(gen.randint(1,4))])
                example,label = generate_color_detection_train_example(text,size=(100,200),background=gen.choice(backgrounds),generator=gen,font_file="fonts/reiko.ttf")
                to_eval = example.copy()
                to_eval = transform_sample(to_eval).unsqueeze(0).type(torch.FloatTensor).to(pytorch_device)
                results = model(to_eval)[0]
                color = np.array(results.cpu().numpy() * 255,dtype=np.int32)
                print("Detected color",color)
                print("Actual color",label)
                display_image(example,"Test Frame")
            except KeyboardInterrupt:
                break