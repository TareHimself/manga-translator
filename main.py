import argparse
import cv2
import numpy as np
from mt.pipelines import FullConversion
from mt.translators import HelsinkiNlpJapaneseToEnglish
from mt.ocr import MangaOcr
import sys

print(sys.platform)


def run_live():
    converter = FullConversion()

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0)
        while True:
            et, imgBase = cap.read()
            if et:
                frame = imgBase
                frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB),
                    (int(1920 / 2), int(1080 / 2)),
                )

                frame = converter([frame])[0]

                if frame is not None:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)

    else:
        from mss import mss

        with mss() as sct:
            monitor = sct.monitors[1]
            while True:
                frame = np.array(sct.grab(monitor))
                frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB),
                    (int(1920 / 1), int(1080 / 1)),
                )

                frame = converter([frame])[0]

                if frame is not None:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)


def do_convert(files: list[str]):
    converter = FullConversion()
    filenames = files
    converted = converter([cv2.imread(file) for file in filenames])

    for filename, frame in zip(filenames, converted):
        ext = filename.split(".")[1]
        cv2.imwrite(
            filename[0 : len(filename) - (len(ext) + 1)] + "_converted." + ext,
            frame,
        )


def main():
    parser = argparse.ArgumentParser(
        prog="Manga Translator",
        description="Translates Manga Chapters",
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["live", "convert"],
        required=True,
        help="What mode to run",
    )
    parser.add_argument("-f", "--files", nargs="+", help="A list of images to convert")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
    else:
        if args.mode == "live":
            run_live()
        elif args.mode == "convert":
            if args.files is None:
                parser.print_help()
            else:
                do_convert(args.files)


if __name__ == "__main__":
    main()
