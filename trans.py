from translator.pipelines import FullConversion
from translator.ocr.manga import MangaOcr
from translator.translators.deepl import DeepLTranslator
from translator.translators.openai import OpenAiTranslator
import os
import math
import cv2
import re

EXTENSION_REGEX = r".*\.([a-zA-Z0-9]+)"
def do_convert(files: list[str],batch_size = 8):
    converter = FullConversion(
        translator=OpenAiTranslator(api_key="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"),
        ocr=MangaOcr(),
        color_detect_model=None
    )
    filenames = files
    batches = math.ceil(len(filenames) / 8)
    for i in range(batches):
        files_to_convert = filenames[i * batch_size: (i + 1) * batch_size]
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

base_path = f"D:\\Github\\manga-translator\\trans-test"
files = list(map(lambda a: [base_path,a,os.path.join(base_path,a)],os.listdir(base_path)))
# print("FILES",files)
do_convert(files)