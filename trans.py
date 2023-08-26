from translator.core.pipelines import FullConversion
from translator.core.ocr import MangaOcr
from translator.core.translators import DeepLTranslator
import os
import math
import cv2
import re
EXTENSION_REGEX = r".*\.([a-zA-Z0-9]+)"
def do_convert(files: list[str]):
    converter = FullConversion(
        translator=DeepLTranslator(auth_token="0e0edc82-62cd-d0d5-422b-004d07e029ef:fx"),
        ocr=MangaOcr(),
    )
    filenames = files
    batches = math.ceil(len(filenames) / 4)
    for i in range(batches):
        files_to_convert = filenames[i * 4: (i + 1) * 4]
        print("CONVERTING",files_to_convert)
        for filename, data in zip(
                files_to_convert, converter([cv2.imread(file[2]) for file in files_to_convert])
        ):
            frame = data
            ext = re.findall(EXTENSION_REGEX, filename[2])[0]
            cv2.imwrite(
                os.path.join(filename[0],"CONVERTED_" + filename[1][0: len(filename[1]) - (len(ext) + 1)] + "." + ext),
                frame,
            )
        print(f"Converted Batch {i + 1}/{batches}")

base_path = f"D:\\Github\\manga-translator\\exa\\Erotic_x_Anabolic_v03\\Erotic x Anabolic v03"
files = list(map(lambda a: [base_path,a,os.path.join(base_path,a)],os.listdir(base_path)))
# print("FILES",files)
do_convert(files)

# do_convert(["D:\\Github\\manga-translator\\exa\\Erotic_x_Anabolic_v04\\Erotic x Anabolic v04\\0001.jpg"])