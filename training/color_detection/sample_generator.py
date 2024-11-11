import os
from faker import Faker
from tqdm import tqdm
from html2image import Html2Image


def make_html_str(content: str,font_file: str,content_color: str,background_color: str):
    return """<!DOCTYPE html>
                    <html>
                    <head>
                    <style>
                    @font-face {
                        font-family: customFont;
                        src:url("@ifont");
                    }

                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        align-content: center;
                        background: @ipcolor;
                        color: @iscolor;
                    }

                    h1{
                        font-family: customFont;
                        text-align: center;
                        font-size: 18px;
                    }
                    </style>
                    </head>
                    <body>
                        <h1>@icontent</h1>
                    </body>
                    </html>""".replace("@ifont",os.path.abspath(font_file).replace("\\","/")).replace("@ipcolor",background_color).replace("@iscolor",content_color).replace("@icontent",content)
class SampleGenerator:
    def __init__(self,cache_path: str = "./datasets") -> None:
        self.cache_path = cache_path


    def run(self,num_samples: int,languages: list[str] = ["en_US"],seed: int = 20) -> str:
        out_dir = os.path.join(os.path.abspath(self.cache_path),f"dataset_{seed}_{num_samples}")
        if os.path.exists(out_dir):
            return out_dir
        
        try:
            hti = Html2Image(output_path=out_dir,disable_logging=True)
            try:
                os.makedirs(out_dir)
            except OSError:
                pass

            faker_instances = [Faker(lang, use_weighting=False) for lang in languages]
            for x in faker_instances:
                x.seed_instance(seed)


            num_fakers = len(faker_instances)

            for i in tqdm(range(num_samples), desc="Generating Phrases"):
                faker_index = (i + 1) % num_fakers
                faker_instance = faker_instances[faker_index]
                color_index = (i + 1) % 2
                color_fg = "255_255_255" if color_index == 0 else "0_0_0"
                color_bg = "0_0_0" if color_index == 0 else "255_255_255"
                html = make_html_str(faker_instance.name(),"D:\\Github\\manga-translator\\fonts\\animeace2_bld.ttf",f"rgb({color_fg.replace('_',',')})",f"rgb({color_bg.replace('_',',')})")
                hti.screenshot(html_str=html, save_as=f"{i}_{color_fg}_255_255_255.png",size=(200,200))
                #self.make_sample(hti,os.path.join(out_dir,,[200,200])

            return out_dir
        except:
            os.removedirs(out_dir)
            raise

        



        
