from typing import Any
from ultralytics import YOLO
import cv2
import numpy as np
from manga_ocr import MangaOcr
import sys
import requests
import traceback
from requests.utils import requote_uri
from utils import extract_bubble, draw_text_in_bubble, debug_image, cv2_to_pil
from translators import Translator


class FullConversion:
    def __init__(
        self,
        detect_model="models/detection.pt",
        seg_model="models/segmentation.pt",
        translator=Translator(),
        debug=False,
    ) -> None:
        self.segmentation_model = YOLO(seg_model)
        self.detection_model = YOLO(detect_model)
        self.translator = translator
        self.debug = debug

    def filter_results(self, results, min_confidence=0.65):
        bounding_boxes = np.array(results.boxes.xyxy.cpu(), dtype="int")

        classes = np.array(results.boxes.cls.cpu(), dtype="int")

        confidence = np.array(results.boxes.conf.cpu(), dtype="float")

        filtered = []
        for box, obj_class, conf in zip(bounding_boxes, classes, confidence):
            if conf >= min_confidence:
                has_similar = False
                for item, _, __ in filtered:
                    diff = np.average(np.absolute(item - box))
                    if diff < 10:
                        has_similar = True
                        break
                if has_similar:
                    continue
                filtered.append((box, obj_class, conf))

        return filtered

    def __call__(self, frames: np.ndarray):
        device = 0 if sys.platform != "darwin" else "mps"
        processed = []
        for result, frame in zip(self.detection_model(frames, device=device), frames):
            mask = np.zeros_like(frame, dtype=frame.dtype)

            segmentation_results = self.segmentation_model(frame, device=device)[0]

            if segmentation_results.masks is not None:
                for seg in list(
                    map(lambda a: a.astype("int"), segmentation_results.masks.xy)
                ):
                    cv2.fillPoly(mask, [seg], (255, 255, 255))

            filtered = self.filter_results(result)

            for box, cls, conf in filtered:
                # if conf < 0.65:
                #     continue

                # print(get_ocr(get_box_section(frame, box)))
                color = (0, 0, 255) if cls == 1 else (0, 255, 0)

                (x1, y1, x2, y2) = box

                if cls == 1:
                    bubble = frame[y1:y2, x1:x2]
                    text_mask = mask[y1:y2, x1:x2]
                    cleaned, text_as_image, bubble_mask = extract_bubble(
                        bubble, text_mask
                    )
                    frame[y1:y2, x1:x2] = cleaned

                    if self.translator:
                        translation = self.translator(text_as_image)
                        frame[y1:y2, x1:x2] = draw_text_in_bubble(
                            bubble, bubble_mask, translation
                        )
                    else:
                        frame[y1:y2, x1:x2] = draw_text_in_bubble(bubble, bubble_mask)

                if self.debug:
                    cv2.putText(
                        frame,
                        str(f"{result.names[cls]} | {conf * 100:.1f}%"),
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        2,
                    )

            processed.append(frame)
        return processed
