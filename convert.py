import os
import sys
import cv2
from pipelines import FullConversion
from translators import DeepLTranslator

if len(sys.argv) < 2:
    raise Exception("No file's specified")

converter = FullConversion(translator=DeepLTranslator(auth_token=""))

filenames = sys.argv[1 : len(sys.argv)]
filenames = sys.argv[1 : len(sys.argv)]
converted = converter([cv2.imread(file) for file in filenames])

for filename, frame in zip(filenames, converted):
    ext = filename.split(".")[1]
    cv2.imwrite(
        filename[0 : len(filename) - (len(ext) + 1)] + "_converted." + ext, frame
    )
