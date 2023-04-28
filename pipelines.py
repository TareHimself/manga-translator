from ultralytics import YOLO
import cv2
import numpy as np
from manga_ocr import MangaOcr
import requests
import traceback
from requests.utils import requote_uri
from utils import extract_bubble, draw_text_in_bubble, debug_image, cv2_to_pil


class FullConversion:
    def __init__(
        self,
        detect_model="models/detection.pt",
        seg_model="models/segmentation.pt",
        translator_auth=None,
        debug=False,
    ) -> None:
        self.segmentation_model = YOLO(seg_model)
        self.detection_model = YOLO(detect_model)
        self.ocr = MangaOcr()
        self.translator_auth = translator_auth
        self.debug = debug

    def translate_text(self, text):
        if self.translator_auth is None or len(self.translator_auth) == 0:
            return "Need DeepL Auth"

        try:
            data = [("target_lang", "EN-US"), ("source_lang", "JA")]
            data.append(("text", text))
            uri = f"https://api-free.deepl.com/v2/translate?{'&'.join([f'{data[i][0]}={data[i][1]}' for i in range(len(data))])}"
            uri = requote_uri(uri)
            return requests.post(
                uri,
                headers={
                    "Authorization": f"DeepL-Auth-Key {self.translator_auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            ).json()["translations"][0]["text"]
        except Exception as e:
            traceback.print_exc()
            return "Failed To Get Translation"

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

    def __call__(self, frames: np.array):
        processed = []
        for result, frame in zip(self.detection_model(frames, device=0), frames):
            mask = np.zeros_like(frame, dtype=frame.dtype)

            segmentation_results = self.segmentation_model(frame, device=0)[0]

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
                    cleaned, text, bubble_mask = extract_bubble(bubble, text_mask)
                    frame[y1:y2, x1:x2] = cleaned

                    if (
                        self.translator_auth is not None
                        and len(self.translator_auth) > 0
                    ):
                        text_to_translate = self.ocr(cv2_to_pil(text))
                        translated = self.translate_text(text_to_translate)
                        frame[y1:y2, x1:x2] = draw_text_in_bubble(
                            bubble, bubble_mask, translated
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
