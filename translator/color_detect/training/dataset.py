import random
from threading import Thread, Event
from faker import Faker
from tqdm import tqdm
from torch.utils.data import Dataset
import math
import numpy as np
import os
import cv2
from .constants import IMAGE_SIZE
from translator.color_detect.training.utils import (
    generate_color_detection_train_example,
    apply_transforms,
    lab_to_rgb_np,
    rgb_to_lab_np,
    normalize_lab_np,
    rgb_to_hsv,
    normalize_hsv,
    denormalize_hsv,
    hsv_to_rgb,
    denormalize_lab_np,
)

def process_label(label):
    result = np.zeros_like(label)
    color_lab = rgb_to_lab_np(np.array([label[:3], label[3:6]]))
    color_lab = normalize_lab_np(color_lab)
    result[:6] = np.concatenate(color_lab,axis=0)
    result[-1] = label[-1]
    return result[:3]

class ColorDetectionDataset(Dataset):


    def __init__(
        self,
        generate_target=0,
        generate_size: tuple[int, int] = (IMAGE_SIZE * 2, IMAGE_SIZE * 2),
        min_generate_size: tuple[int, int] = (round(IMAGE_SIZE) / 2, round(IMAGE_SIZE) / 2),
        backgrounds: list = [],
        generator_seed=0,
        languages: list[str] = ["en_US"],
        fonts: list[list[str]] = [
            [
                "fonts/Roboto-MediumItalic.ttf",
                "fonts/Roboto-Regular.ttf",
                "fonts/Roboto-Thin.ttf",
                "fonts/Roboto-ThinItalic.ttf",
                "fonts/Roboto-Black.ttf",
                "fonts/Roboto-BlackItalic.ttf",
                "fonts/Roboto-Bold.ttf",
                "fonts/Roboto-BoldItalic.ttf",
                "fonts/Roboto-Italic.ttf",
                "fonts/Roboto-Light.ttf",
                "fonts/Roboto-LightItalic.ttf",
                "fonts/Roboto-Medium.ttf",
            ],
            [
                "fonts/NotoSansJP-ExtraLight.ttf",
                "fonts/NotoSansJP-Light.ttf",
                "fonts/NotoSansJP-Medium.ttf",
                "fonts/NotoSansJP-Regular.ttf",
                "fonts/NotoSansJP-SemiBold.ttf",
                "fonts/NotoSansJP-Thin.ttf",
                "fonts/NotoSansJP-Black.ttf",
                "fonts/NotoSansJP-Bold.ttf",
                "fonts/NotoSansJP-ExtraBold.ttf",
            ],
        ],
        num_workers=12,
        cache_dir = None
    ) -> None:
        self.saved_to = os.path.join(cache_dir,f"seed-{generator_seed}") if cache_dir is not None else None
        self.images_path = os.path.join(self.saved_to,"images") if self.saved_to is not None else None
        self.labels_path = os.path.join(self.saved_to,"labels") if self.saved_to is not None else None
        self.generate_target = generate_target
        self.generate_size = generate_size
        self.examples = []
        self.labels = []
        self.languages = languages
        self.fonts = [[os.path.abspath(y) for y in x] for x in fonts]

        num_generated = 0

        if self.saved_to is not None:
            os.makedirs(self.images_path,exist_ok=True)
            os.makedirs(self.labels_path,exist_ok=True)

        if self.images_path is not None and os.path.exists(self.images_path):
            for file in tqdm(os.listdir(self.images_path), desc="Loading generated data"):
                if file.endswith(".png"):
                    idx = int(file.split(".")[0])
                    image = cv2.imread(os.path.join(self.images_path,file))
                    label_tx = ""
                    with open(os.path.join(self.labels_path,f"{idx}.txt"),"r") as f:
                        label_tx = f.readline()
                    label = np.array([int(x) for x in label_tx.split(",")]).astype(np.float32)
                    label = process_label(label)
                    self.examples.append(image)
                    self.labels.append(label)
                    num_generated += 1

        backgrounds.append(np.zeros((*generate_size[::-1], 3), dtype=np.uint8))
        backgrounds.append(np.full((*generate_size[::-1], 3), 255, dtype=np.uint8))
        generator = random.Random(generator_seed)
        np.random.seed(generator_seed)

        faker_instances = [Faker(lang, use_weighting=False) for lang in self.languages]
        for x in faker_instances:
            x.seed_instance(generator_seed)

        num_fakers = len(faker_instances)

        
        stop_threads_event = Event()

        faker_phrases = []
        to_generate = generate_target - num_generated

        if to_generate > 0:
            for i in tqdm(range(to_generate), desc="Generating Phrases"):
                faker_index = (i + 1) % num_fakers

                faker_phrases.append(
                    faker_instances[faker_index].name()
                    # " ".join(
                    #     [
                    #         faker_instances[faker_index].name()
                    #         for x in range(generator.randint(1, 5))
                    #     ]
                    # )
                )  # phrase is made up of names)

        
            loader = tqdm(total=to_generate, desc="Generating Samples")

            def make_sample(start,phrases):
                nonlocal num_generated
                for idx in range(len(phrases)):
                    phrase = phrases[idx]
                    abs_idx = start + idx
                    if stop_threads_event.is_set():
                        break

                    example, example_color = generate_color_detection_train_example(
                        phrase,
                        background=generator.choice(
                            backgrounds
                            + [
                                (np.random.rand(*generate_size[::-1], 3) * 255).astype(
                                    np.uint8
                                )
                            ]
                        ),
                        size=[
                            (
                                generator.randrange(min_generate_size[0], generate_size[0])
                                if min_generate_size[0] != generate_size[0]
                                else generate_size[0]
                            ),
                            (
                                generator.randrange(min_generate_size[0], generate_size[1])
                                if min_generate_size[1] != generate_size[1]
                                else generate_size[1]
                            ),
                        ],
                        font_file=generator.choice(self.fonts[faker_index]),
                        # shift_max=max_text_offset,
                        generator=generator,
                        force_outline= True if abs_idx % 2 == 0 else False
                    )

                    if self.saved_to is not None:
                        cv2.imwrite(os.path.join(self.images_path,f"{abs_idx}.png"),example)
                        with open(os.path.join(self.labels_path,f"{abs_idx}.txt"),"w") as f:
                            f.write(",".join([f"{x}" for x in example_color.tolist()]))
                    # print(example_color)
                    # display_image(example,"SAMPLE")
                    # print(label,phrase)
                    # debug_image(example,"Generated sample")
                    # executor.submit(add_sample,label,example)

                    self.examples.append(example)
                    label = example_color.astype(np.float32)
                    label = process_label(label)
                    self.labels.append(label)
                    loader.update()
                    num_generated += 1

            amount_per_section = math.ceil(len(faker_phrases) / num_workers)
            sections = []
            for x in range(num_workers):
                sections.append([num_generated + (x * amount_per_section),faker_phrases[x * amount_per_section : (x + 1) * amount_per_section]])

            pending = [
                Thread(target=make_sample, daemon=True, group=None, args=section)
                for section in sections
            ]

            for p in pending:
                p.start()

            while True in [x.is_alive() for x in pending]:
                for p in pending:
                    try:
                        p.join(timeout=0.5)
                    except KeyboardInterrupt:
                        print("SHUTTING DOWN")
                        stop_threads_event.set()
                        break

                if stop_threads_event.is_set():
                    break

        for i in tqdm(range(len(self.examples)), desc="Resizing Samples"):
            self.examples[i] = apply_transforms(self.examples[i]).numpy()

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
