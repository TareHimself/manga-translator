from manga_translator.pipelines.image_to_image import ImageToImagePipeline
from manga_translator.pipelines.cbz import CbzPipeline
from manga_translator.core.plugin import ColorDetector
from manga_translator.cleaning.opencv import OpenCvCleaner
from manga_translator.detection.yolo import YoloDetector
from manga_translator.segmentation.yolo import YoloSegmenter
from manga_translator.translation.deepl import DeepLTranslator
from manga_translator.ocr.manga_ocr import MangaOCR
from manga_translator.drawing.horizontal import HorizontalDrawer
import asyncio
import cv2
import torch
def main():
    device = torch.device('cpu')
    pipeline = ImageToImagePipeline(
        translator=DeepLTranslator(auth_key="Some deepl api key"),
        detector=YoloDetector(r"Some yolo detector path",device=device),
        segmenter=YoloSegmenter(r"Some yolo segmenter path",device=device),
        cleaner=OpenCvCleaner(),#AllWhiteCleaner(),
        drawer=HorizontalDrawer(font_file=r"Some Font File"),
        ocr=MangaOCR(device=device), #MangaOcr(device=torch.device('cuda:0')),
        color_detector=ColorDetector(),
    )
    image = cv2.imread(r"./file.png")
    result = asyncio.run(pipeline([image]))[0]
    cv2.imshow("Translated",result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
