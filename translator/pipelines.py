import time
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from translator.utils import (
    display_image,
    mask_text_and_make_bubble_mask,
    get_bounds_for_text,
    TranslatorGlobals,
    has_white,
    get_model_path,
    apply_mask
)
from translator.color_detect.utils import apply_transforms
import traceback
import threading
import torch
import asyncio
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from translator.color_detect.models import get_color_detection_model
from translator.core.plugin import Drawable, Translator, Ocr, Drawer, Cleaner
from translator.cleaners.deepfillv2 import DeepFillV2Cleaner
from translator.drawers.horizontal import HorizontalDrawer


# def inpaint(image, mask, radius=2, iterations=3):
#     result = image
#     for i in range(iterations):
#         result = cv2.inpaint(
#             result,
#             mask,
#             radius,
#             cv2.INPAINT_NS,
#         )
#     return result

TranslatorGlobals.COLOR_BLACK


def resize_percent(image, dest_percent=50):
    width = int(image.shape[1] * dest_percent / 100)
    height = int(image.shape[0] * dest_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class FullConversion:
    def __init__(
        self,
        detect_model: str = get_model_path("detection.pt"),
        seg_model: str = get_model_path("segmentation.pt"),
        color_detect_model: Union[str, None] = get_model_path("color_detection.pt"), # Performance is not where I would like it to be at,
        translator: Translator = Translator(),
        ocr: Ocr = Ocr(),
        drawer: Drawer = HorizontalDrawer(),
        cleaner: Cleaner = DeepFillV2Cleaner(),
        translate_free_text: bool = False,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        yolo_device=0 if torch.cuda.is_available() else "cpu",
        debug=False,
    ) -> None:
        self.device = device
        print("Pipeline created using",device)
        self.yolo_device = yolo_device
        self.segmentation_model = YOLO(seg_model)
        self.detection_model = YOLO(detect_model)
        try:
            if color_detect_model is not None:
                self.color_detect_model = get_color_detection_model(
                    weights_path=color_detect_model, device=self.device
                )
                self.color_detect_model.eval()
            else:
                self.color_detect_model = None
        except:
            self.color_detect_model = None
            traceback.print_exc()

        self.translate_free_text = translate_free_text
        self.translator = translator
        self.ocr = ocr
        self.drawer = drawer
        self.debug = debug
        self.cleaner = cleaner
        self.frame_process_mutex = threading.Lock()

    def filter_results(self, results, min_confidence=0.1):
        bounding_boxes = np.array(results.boxes.xyxy.cpu(), dtype="int")

        classes = np.array(results.boxes.cls.cpu(), dtype="int")

        confidence = np.array(results.boxes.conf.cpu(), dtype="float")

        raw_results: list[tuple[tuple[int, int, int, int], str, float]] = []

        # has_similar = False
        #         for item, _, __ in filtered:
        #             print(np.absolute(item - box))
        #             diff = np.average(np.absolute(item - box))
        #             if diff < 10:
        #                 has_similar = True
        #                 break
        #         if has_similar:
        #             continue
        
        for box, obj_class, conf in zip(bounding_boxes, classes, confidence):
            if conf >= min_confidence:
                raw_results.append((box, results.names[obj_class], conf))

        raw_results.sort(key= lambda a: 1 - a[2])

        results: list[tuple[tuple[int, int, int, int], str, float]] = []

        # print(f"Starting with {len(raw_results)} results")
        # while len(raw_results) > 0:
        #     results.append(raw_results[0])
        #     raw_results = list(filter(lambda a: iou(raw_results[0][0],a[0]) < 0.5,raw_results))

        results = raw_results

        # print(f"Ended with {len(results)} results")
        return results

    async def process_ml_results(self, detect_result, seg_result, frame):
        text_mask = np.zeros_like(frame, dtype=frame.dtype)

        if seg_result.masks is not None:  # Fill in segmentation results
            for seg in list(map(lambda a: a.astype("int"), seg_result.masks.xy)):
                cv2.fillPoly(text_mask, [seg], (255, 255, 255))

        detect_result = self.filter_results(detect_result)

        for bbox, cls, conf in detect_result:  # fill in text free results
            if cls == "text_free":
                (x1, y1, x2, y2) = bbox
                text_mask = cv2.rectangle(
                    text_mask, (x1, y1), (x2, y2), (255, 255, 255), -1
                )

        start = time.time()

        frame_clean, text_mask = await self.cleaner(
            frame=frame, mask=text_mask, detection_results=detect_result
        )  # segmentation_results.boxes.xyxy.cpu().numpy()

        print(f"Inpainting => {time.time() - start} seconds")

        return frame, frame_clean, text_mask, detect_result

    async def process_frame(self, detect_result, seg_result, input_frame):
        try:
            frame, frame_clean, text_mask, detect_result = await self.process_ml_results(
                detect_result, seg_result, input_frame
            )

            to_translate = []
            # First pass, mask all bubbles
            for bbox, cls, conf in detect_result:
                try:
                    # if conf < 0.65:
                    #     continue

                    # print(get_ocr(get_box_section(frame, box)))
                    color = (0, 0, 255) if cls == 1 else (0, 255, 0)

                    (x1, y1, x2, y2) = bbox

                    class_name = cls

                    bubble = frame[y1:y2, x1:x2]
                    bubble_clean = frame_clean[y1:y2, x1:x2]
                    bubble_text_mask = text_mask[y1:y2, x1:x2]

                    if class_name == "text_bubble":
                        if has_white(bubble_text_mask):
                            text_only, bubble_mask = mask_text_and_make_bubble_mask(
                                bubble, bubble_text_mask, bubble_clean
                            )

                            frame[y1:y2, x1:x2] = bubble_clean
                            text_draw_bounds = get_bounds_for_text(bubble_mask)

                            pt1, pt2 = text_draw_bounds

                            pt1_x, pt1_y = pt1
                            pt2_x, pt2_y = pt2

                            pt1_x += x1
                            pt2_x += x1
                            pt1_y += y1
                            pt2_y += y1

                            to_translate.append([(pt1_x, pt1_y, pt2_x, pt2_y), text_only])

                            # frame = cv2.rectangle(frame,(x1,y1),(x2,y2),color=(255,255,0),thickness=2)
                            # debug_image(text_only,"Text Only")
                    else:
                        if self.translate_free_text:
                            free_text = frame[y1:y2, x1:x2]
                            if has_white(free_text):
                                text_only, _ = mask_text_and_make_bubble_mask(
                                    free_text, bubble_text_mask, bubble_clean
                                )

                                to_translate.append([(x1, y1, x2, y2), text_only])

                            frame[y1:y2, x1:x2] = frame_clean[y1:y2, x1:x2]
                        else:
                            frame[y1:y2, x1:x2] = frame_clean[y1:y2, x1:x2]

                    if self.debug:
                        cv2.putText(
                            frame,
                            str(f"{cls} | {conf * 100:.1f}%"),
                            (x1, y1 - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            color,
                            2,
                        )
                except:
                    traceback.print_exc()

            # second pass, fix intersecting text areas
            # for i in range(len(to_translate)):
            #     bbox_a = to_translate[i][0]
            #     text_bounds_a_local = to_translate[i][3]
            #     text_bounds_a = [
            #         [
            #             text_bounds_a_local[0][0] + bbox_a[0],
            #             text_bounds_a_local[0][1] + bbox_a[1],
            #         ],
            #         [
            #             text_bounds_a_local[1][0] + bbox_a[0],
            #             text_bounds_a_local[1][1] + bbox_a[1],
            #         ],
            #     ]
            #     for x in range(len(to_translate)):
            #         if x == i:
            #             continue

            #         bbox_b = to_translate[x][0]
            #         text_bounds_b_local = to_translate[x][3]
            #         text_bounds_b = [
            #             [
            #                 text_bounds_b_local[0][0] + bbox_b[0],
            #                 text_bounds_b_local[0][1] + bbox_b[1],
            #             ],
            #             [
            #                 text_bounds_b_local[1][0] + bbox_b[0],
            #                 text_bounds_b_local[1][1] + bbox_b[1],
            #             ],
            #         ]

            #         fix_result = fix_intersection(
            #             text_bounds_a[0],
            #             text_bounds_a[1],
            #             text_bounds_b[0],
            #             text_bounds_b[1],
            #         )
            #         found_intersection = fix_result[4]
            #         if found_intersection:
            #             to_translate[i][3] = [
            #                 [
            #                     fix_result[0][0] - bbox_a[0],
            #                     fix_result[0][1] - bbox_a[1],
            #                 ],
            #                 [torch.cuda.is_available()
            #                     fix_result[1][0] - bbox_a[0],
            #                     fix_result[1][1] - bbox_a[1],
            #                 ],
            #             ]
            #             to_translate[x][3] = [
            #                 [
            #                     fix_result[2][0] - bbox_b[0],
            #                     fix_result[2][1] - bbox_b[1],
            #                 ],
            #                 [
            #                     fix_result[3][0] - bbox_b[0],
            #                     fix_result[3][1] - bbox_b[1],
            #                 ],
            #             ]
            # print(
            #     bbox_a,
            #     text_bounds_a_local,
            #     text_bounds_a,
            #     "\n",
            #     bbox_b,
            #     text_bounds_b_local,
            #     text_bounds_b,
            # )
            # debug_image(to_translate[i][1])
            # debug_image(to_translate[x][1])
            # cv2.rectangle(
            #     frame,
            #     fix_result[0],
            #     fix_result[1],
            #     (0, 255, 255),
            #     1,
            # )
            # cv2.rectangle(
            #     frame,
            #     fix_result[2],
            #     fix_result[3],
            #     (0, 255, 255),
            #     1,
            # )
            # print("intersection found")

            # third pass, draw text
            draw_colors = [(TranslatorGlobals.COLOR_BLACK,TranslatorGlobals.COLOR_BLACK,False) for x in to_translate]

            start = time.time()
            if self.color_detect_model is not None and len(draw_colors) > 0:
                with torch.no_grad():  # model needs work
                    with torch.inference_mode():
                        with self.frame_process_mutex:  # this may not be needed

                            images = [apply_transforms(frame_with_text.copy()) for _, frame_with_text in to_translate]

                            draw_colors = [((y[0:3] * 255).astype(np.uint8),(y[3:-1] * 255).astype(np.uint8),(True if y[-1] > 0.5 else False)) for y in [
                                x.cpu().numpy()
                                for x in self.color_detect_model(
                                    torch.stack(images).to(
                                        self.device
                                    )
                                )
                            ]]
            else:
                print("Using black since color detect model is 'None'")

            print(f"Color Detection => {time.time() - start} seconds")

            start = time.time()

            to_draw = []

            if self.translator and self.ocr and len(to_translate) > 0:
                bboxes,images = zip(*to_translate)

                ocr_results = await self.ocr(list(images))

                translation_results = await self.translator(ocr_results)

                to_draw = []
                for bbox,translation,color in zip(bboxes,translation_results,draw_colors):

                    (x1, y1, x2, y2) = bbox
                    draw_area = frame[y1:y2, x1:x2].copy()

                    to_draw.append(Drawable(color=color,frame=draw_area,translation=translation))

                    

                print(f"Ocr And Translation => {time.time() - start} seconds")

                start = time.time()

                drawn_frames = await self.drawer(to_draw)


                for bbox, drawn_frame in zip(bboxes,drawn_frames):
                    (x1, y1, x2, y2) = bbox
                    drawn_frame,drawn_frame_mask = drawn_frame
                    frame[y1:y2, x1:x2] = apply_mask(drawn_frame,frame[y1:y2, x1:x2],drawn_frame_mask)

                    

                print(f"Drawing => {time.time() - start} seconds")
            return frame
        except:
            traceback.print_exc()
            return input_frame

    async def __call__(
        self,
        images: list[np.ndarray],
    ) -> list[np.ndarray]:
        # frames = [resize_percent(x, 50) for x in frames]
        total_start = time.time()
        start = time.time()
        to_process = [
            x
            for x in zip(
                self.detection_model(images, device=self.yolo_device, verbose=False),
                self.segmentation_model(images, device=self.yolo_device, verbose=False),
                images,
            )
        ]

        print(f"Yolov8 Models => {time.time() - start} seconds")

        tasks = [self.process_frame(detect_result=detect_result,seg_result=seg_result,input_frame=frame) for detect_result, seg_result, frame in to_process]
        results = await asyncio.gather(*tasks)

        print(f"Total Process => {time.time() - total_start} seconds")
        return results