
import random
import numpy as np
from faker import Faker
from tqdm import tqdm
from torch.utils.data import Dataset
from ..utils import generate_color_detection_train_example, prep_color_detection_sample

class ColorDetectionDataset(Dataset):
    def __init__(self,generate_target = 0,generate_size: tuple[int,int] = (224,224),backgrounds=[],generator_seed=0) -> None:
        self.generate_target = generate_target
        self.generate_size = generate_size
        self.examples = []
        self.labels = []
        has_extra_backgrounds = len(backgrounds) > 0
        generator = random.Random(generator_seed)

        fake_ja = Faker(['ja_JP'])
        fake_ja.seed_instance(generator_seed)

        fake_en = Faker(['en_US'])
        fake_en.seed_instance(generator_seed)

        for i in tqdm(range(generate_target),desc="Generating Dataset"):
            is_ja = (i + 1) % 2 == 0
            phrase  = (fake_ja if is_ja else fake_en).name()

            background_choices = [np.ones((*generate_size,3),dtype=np.uint8) * 255,np.ones((*generate_size,3),dtype=np.uint8) * 0]

            if has_extra_backgrounds:
                background_choices.append(generator.choice(backgrounds))

            example,label = generate_color_detection_train_example(phrase,background=generator.choice(background_choices),size=generate_size,font_file=generator.choice(["fonts/reiko.ttf","fonts/msmincho.ttf"]) if is_ja else generator.choice(["fonts/animeace2_reg.ttf","fonts/BlambotClassicBB.ttf"]),generator=generator)
            
            # print(label,phrase)
            # debug_image(example,"Generated sample")

            self.examples.append(prep_color_detection_sample(example).numpy())
            self.labels.append(label / 255)

        

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)