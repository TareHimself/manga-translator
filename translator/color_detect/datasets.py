import random
from threading import Thread, Event
from faker import Faker
from tqdm import tqdm
from torch.utils.data import Dataset
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
from ..utils import generate_color_detection_train_example, transform_sample


class ColorDetectionDataset(Dataset):
    def __init__(
        self,
        generate_target=0,
        generate_min_size: tuple[int, int] = (224, 224),
        generator_size_delta: int = 300,
        max_text_offset: tuple[int, int] = (20, 20),
        backgrounds=[],
        generator_seed=0,
        languages: list[str] = ["ja_JP", "en_US"],
        fonts: list[list[str]] = [
            ["fonts/reiko.ttf", "fonts/msmincho.ttf"],
            ["fonts/animeace2_reg.ttf", "fonts/BlambotClassicBB.ttf"],
        ],
        num_workers=5,
    ) -> None:
        self.generate_target = generate_target
        self.generate_size = generate_min_size
        self.examples = []
        self.labels = []
        self.languages = languages
        self.fonts = fonts
        has_extra_backgrounds = len(backgrounds) > 0
        generator = random.Random(generator_seed)

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

                example, label = generate_color_detection_train_example(
                    phrase,
                    background=generator.choice(backgrounds)
                    if has_extra_backgrounds
                    else None,
                    size=[
                        generator.randint(x, x + generator_size_delta)
                        for x in generate_min_size
                    ],
                    font_file=generator.choice(self.fonts[faker_index]),
                    shift_max=max_text_offset,
                    generator=generator,
                )

                # print(label,phrase)
                # debug_image(example,"Generated sample")
                # executor.submit(add_sample,label,example)
                self.examples.append(example)
                self.labels.append(label / 255)
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
            self.examples[i] = transform_sample(self.examples[i]).numpy()

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
