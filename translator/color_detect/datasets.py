import random
from threading import Thread, Event
from faker import Faker
from tqdm import tqdm
from torch.utils.data import Dataset
import math
import numpy as np

from translator.color_detect.constants import IMAGE_SIZE
from translator.utils import display_image
from .utils import generate_color_detection_train_example, apply_transforms


class ColorDetectionDataset(Dataset):
    def __init__(
        self,
        generate_target=0,
        generate_size: tuple[int, int] = (IMAGE_SIZE * 2, IMAGE_SIZE * 2),
        min_generate_size: tuple[int, int] = (round(IMAGE_SIZE / 2), round(IMAGE_SIZE / 2)),
        backgrounds: list =[],
        generator_seed=0,
        languages: list[str] = ["en_US"],
        fonts: list[list[str]] = [
            [
    'fonts/Roboto-MediumItalic.ttf',
    'fonts/Roboto-Regular.ttf',
    'fonts/Roboto-Thin.ttf',
    'fonts/Roboto-ThinItalic.ttf',
    'fonts/Roboto-Black.ttf',
    'fonts/Roboto-BlackItalic.ttf',
    'fonts/Roboto-Bold.ttf',
    'fonts/Roboto-BoldItalic.ttf',
    'fonts/Roboto-Italic.ttf',
    'fonts/Roboto-Light.ttf',
    'fonts/Roboto-LightItalic.ttf',
    'fonts/Roboto-Medium.ttf',
],
[
    'fonts/NotoSansJP-ExtraLight.ttf',
    'fonts/NotoSansJP-Light.ttf',
    'fonts/NotoSansJP-Medium.ttf',
    'fonts/NotoSansJP-Regular.ttf',
    'fonts/NotoSansJP-SemiBold.ttf',
    'fonts/NotoSansJP-Thin.ttf',
    'fonts/NotoSansJP-Black.ttf',
    'fonts/NotoSansJP-Bold.ttf',
    'fonts/NotoSansJP-ExtraBold.ttf'
]
        ],
        num_workers=5,
    ) -> None:
        self.generate_target = generate_target
        self.generate_size = generate_size
        self.examples = []
        self.labels = []
        self.languages = languages
        self.fonts = fonts
        backgrounds.append(np.zeros((*generate_size[::-1], 3), dtype=np.uint8))
        backgrounds.append(np.full((*generate_size[::-1], 3), 255, dtype=np.uint8))
        generator = random.Random(generator_seed)
        np.random.seed(generator_seed)

        faker_instances = [Faker(lang, use_weighting=False) for lang in self.languages]
        for x in faker_instances:
            x.seed_instance(generator_seed)

        num_fakers = len(faker_instances)

        num_generated = 0
        stop_threads_event = Event()

        faker_phrases = []

        for i in tqdm(range(generate_target), desc="Generating Phrases"):
            faker_index = (i + 1) % num_fakers

            faker_phrases.append(
                " ".join(
                    [
                        faker_instances[faker_index].name()
                        for x in range(generator.randint(1, 5))
                    ]
                )
            )  # phrase is made up of names)

        loader = tqdm(total=generate_target, desc="Generating Samples")

        def make_sample(*items):
            nonlocal num_generated
            for phrase in items:
                if stop_threads_event.is_set():
                    break

                example, example_color = generate_color_detection_train_example(
                    phrase,
                    background=generator.choice(backgrounds + [(np.random.rand(*generate_size[::-1],3) * 255).astype(np.uint8)]),
                    size=[generator.randrange(min_generate_size[0],generate_size[0]), generator.randrange(min_generate_size[0],generate_size[1])],
                    font_file=generator.choice(self.fonts[faker_index]),
                    # shift_max=max_text_offset,
                    generator=generator,
                )

                # print(example_color)
                # display_image(example,"SAMPLE")
                # print(label,phrase)
                # debug_image(example,"Generated sample")
                # executor.submit(add_sample,label,example)
                
                self.examples.append(example)
                label = example_color.astype(np.float32)
                # label[:-1] = label[:-1] / 255
                label = label / 255
                self.labels.append(label)
                loader.update()
                num_generated += 1

        amount_per_section = math.ceil(len(faker_phrases) / num_workers)
        sections = [
            faker_phrases[x * amount_per_section : (x + 1) * amount_per_section]
            for x in range(num_workers)
        ]

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
