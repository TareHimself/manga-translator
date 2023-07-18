
import random
import concurrent.futures as c_futures
import numpy as np
from threading import Thread
from faker import Faker
from tqdm import tqdm
from torch.utils.data import Dataset
import math
from ..utils import display_image, generate_color_detection_train_example, transform_sample

class ColorDetectionDataset(Dataset):
    def __init__(self,generate_target = 0,generate_min_size: tuple[int,int] = (224,224),generator_size_delta: int = 300,max_text_offset: tuple[int,int] = (20,20),backgrounds=[],generator_seed=0,languages: list[str] = ['ja_JP','en_US'],fonts: list[list[str]] = [["fonts/reiko.ttf","fonts/msmincho.ttf"],["fonts/animeace2_reg.ttf","fonts/BlambotClassicBB.ttf"]] ,num_workers = 5) -> None:
        self.generate_target = generate_target
        self.generate_size = generate_min_size
        self.examples = []
        self.labels = []
        self.languages = languages
        self.fonts = fonts
        has_extra_backgrounds = len(backgrounds) > 0
        generator = random.Random(generator_seed)


        faker_instances = [Faker(lang) for lang in self.languages]
        for x in faker_instances:
            x.seed_instance(generator_seed)

        num_fakers = len(faker_instances)

        loader = tqdm(total=generate_target,desc="Generating Color Detection Dataset")

        def make_sample(items):
            for i in items:
                faker_index = (i + 1) % num_fakers

                phrase  = " ".join([faker_instances[faker_index].name() for x in range(generator.randint(1,5))]) # phrase is made up of names

                example,label = generate_color_detection_train_example(phrase,background=generator.choice(backgrounds) if has_extra_backgrounds else None,size=[generator.randint(x,x + generator_size_delta) for x in generate_min_size],font_file=generator.choice(self.fonts[faker_index]),shift_max=max_text_offset,generator=generator)
                
                # print(label,phrase)
                # debug_image(example,"Generated sample")

                self.examples.append(transform_sample(example).numpy())
                self.labels.append(label / 255)
                loader.update() 

        target = list(range(generate_target))
        ammount_per_section = math.ceil(len(target) / num_workers)
        sections = [target[x * ammount_per_section:(x + 1) * ammount_per_section] for x in range(num_workers)]

        with c_futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

            for section in sections:
                executor.submit(make_sample,items = section)
            
        

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)