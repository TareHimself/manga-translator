import argparse
import cv2
import sys
import os
import math
import re
import numpy as np
from translator.pipelines import FullConversion
from translator.translators.get import get_translators
from translator.ocr.get import get_ocr
from translator.drawers.get import get_drawers

EXTENSION_REGEX = r".*\.([a-zA-Z0-9]+)"


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def convert_to_options_list(classes: list):
    result = ""
    for x in range(len(classes)):
        item = classes[x]
        result += f"{x}) {item.__name__} => {item.__doc__}\n"

    return result[:-1]


def json_to_args(args_str: str):
    args = {}
    for item in args_str.strip().split(","):
        if "=" not in item:
            continue
        a = item.strip()
        equ_idx = a.index("=")
        key = a[0:equ_idx]
        value = eval(a[equ_idx + 1:])
        args[key] = value
    return args


def run_live(tran: int, tran_args: str, ocr: int, ocr_args: str):
    converter = FullConversion(
        translator=get_translators()[tran](**json_to_args(tran_args)),
        ocr=get_ocr()[ocr](**json_to_args(ocr_args)),
    )

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

        scale = 2
        with mss() as sct:
            monitor = sct.monitors[1]
            while True:
                frame = np.array(sct.grab(monitor))
                frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB),
                    (int(1920 / scale), int(1080 / scale)),
                )

                frame = converter([frame])[0]

                if frame is not None:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)


def do_convert(files: list[str], translator: int, translator_args: str, ocr: int, ocr_args: str,drawer: int, drawer_args: str):
    converter = FullConversion(
        translator=get_translators()[translator](**json_to_args(translator_args)),
        ocr=get_ocr()[ocr](**json_to_args(ocr_args)),drawer=get_drawers()[drawer](**json_to_args(drawer_args)),
    )
    filenames = files
    batches = math.ceil(len(filenames) / 4)
    for i in range(batches):
        files_to_convert = filenames[i * 4: (i + 1) * 4]
        for filename, data in zip(
                files_to_convert, converter([cv2.imread(file) for file in files_to_convert])
        ):
            frame = data
            ext = re.findall(EXTENSION_REGEX, filename)[0]
            cv2.imwrite(
                filename[0: len(filename) - (len(ext) + 1)] + "_converted." + ext,
                frame,
            )
        print(f"Converted Batch {i + 1}/{batches}")


def main():
    parser = argparse.ArgumentParser(
        prog="Manga Translator",
        description="Translates Manga Chapters",
        formatter_class=SmartFormatter,
        exit_on_error=True
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["live", "convert"],
        required=True,
        help="What mode to run",
    )

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="A list of images to convert or path to a folder of images",
    )

    parser.add_argument(
        "-o",
        "--ocr",
        default=0,
        type=int,
        help="R|Set the index of the ocr class to use. must be one of the following\n"
             + convert_to_options_list(get_ocr()),
        required=False,
    )

    parser.add_argument(
        "-oa",
        "--ocr-args",
        default="",
        type=str,
        help="Set ocr class args i.e. 'key=value , key2=value'",
        required=False,
    )

    parser.add_argument(
        "-t",
        "--translator",
        default=0,
        type=int,
        help="R|Set the index of the translator class to use. must be one of the following\n"
             + convert_to_options_list(get_translators()),
        required=False,
    )

    parser.add_argument(
        "-ta",
        "--translator-args",
        default="",
        type=str,
        help="Set translator class args i.e. 'key=value , key2=value'",
        required=False,
    )

    parser.add_argument(
        "-dr",
        "--drawer",
        default=0,
        type=int,
        help="R|Set the index of the drawer class to use. must be one of the following\n"
             + convert_to_options_list(get_drawers()),
        required=False,
    )

    parser.add_argument(
        "-dra",
        "--drawer-args",
        default="",
        type=str,
        help="Set drawer class args i.e. 'key=value , key2=value'",
        required=False,
    )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
    else:
        if args.mode == "live":
            run_live(args.tra, args.tra_args, args.ocr, args.ocr_args)
        elif args.mode == "convert":
            if args.files is None:
                parser.print_help()
            else:
                files = args.files
                if len(files) == 1 and os.path.isdir(files[0]):
                    do_convert(
                        [os.path.join(files[0], x) for x in os.listdir(files[0])],
                        args.translator,
                        args.translator_args,
                        args.ocr,
                        args.ocr_args,
                        args.drawer,
                        args.drawer_args,
                    )
                else:
                    do_convert(
                        files,
                        args.translator,
                        args.translator_args,
                        args.ocr,
                        args.ocr_args,
                        args.drawer,
                        args.drawer_args,
                    )


if __name__ == "__main__":
    main()
