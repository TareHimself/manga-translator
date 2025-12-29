import argparse
import asyncio
import os
from typing import Type
import cv2
import yaml
from manga_translator.get import construct_plugin_by_name
from manga_translator.pipelines.image_to_image import ImageToImagePipeline

class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

# def list_to_json()

def write_image(source_path,destination,image):
    dest_name = os.path.basename(source_path)
    cv2.imwrite(os.path.join(destination,dest_name),image)

async def main():
    parser = argparse.ArgumentParser(
        prog="Manga Translator",
        description="Translates Manga Pages",
        formatter_class=SmartFormatter,
        exit_on_error=True
    )

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="A list of images to convert or path to a folder of images",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output directory",
        default="./out"
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=4,
        help="The batch size"
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="The yaml config to use",
        required=True,
    )

    args = parser.parse_args()
    files = args.files
    batch_size: int = args.batch
    output_dir: str = os.path.abspath(args.output)
    files = [os.path.abspath(x) for x in ([os.path.join(files[0], x) for x in os.listdir(files[0])] if len(files) == 1 and os.path.isdir(files[0]) else files)]
    config_file_path = os.path.abspath(args.config)

    os.makedirs(output_dir,exist_ok=True)

    with open(config_file_path,'r') as file:
        data: dict = yaml.safe_load(file)["pipeline"]
        pipeline_args = {}
        for arg_name in data.keys():
            arg_data = data[arg_name]
            if arg_data["class"] == "Default":
                continue

            arg_args = arg_data["args"]
            if arg_args is None:
                arg_args = {}

            pipeline_args[arg_name] = construct_plugin_by_name(arg_data["class"],arg_args)
        
        pipeline = ImageToImagePipeline(**pipeline_args)

        for batch_start in range(0, len(files), batch_size):
            
            print(f"Processing batch [{batch_start} : {batch_start + batch_size}]")
            target_files = files[batch_start : batch_start + batch_size]
            images = await asyncio.gather(
                    *[
                        asyncio.to_thread(cv2.imread,file_path)
                        for file_path in target_files
                    ]
                )
            
            results = await pipeline(images)

            await asyncio.gather(
                    *[
                        asyncio.to_thread(write_image,file_path,output_dir,result)
                        for file_path,result in zip(target_files,results)
                    ]
                )
            print("Done")
            
if __name__ == "__main__":
    asyncio.run(main())
