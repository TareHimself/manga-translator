import cv2
import numpy as np
import sys
from ultralytics import YOLO
from requests.utils import requote_uri
from translator.utils import (
    generate_bubble_mask,
    draw_text_in_bubble,
    get_bounds_for_text,
    fix_intersection,
    inpaint_optimized,
)
from translator.translators import Translator
from translator.ocr import BaseOcr


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


def resize_percent(image, dest_percent=50):
    width = int(image.shape[1] * dest_percent / 100)
    height = int(image.shape[0] * dest_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class FullConversion:
    def __init__(
        self,
        detect_model="models/detection.pt",
        seg_model="models/segmentation.pt",
        translator=Translator(),
        ocr=BaseOcr(),
        debug=False,
    ) -> None:
        self.segmentation_model = YOLO(seg_model)
        self.detection_model = YOLO(detect_model)
        self.translator = translator
        self.ocr = ocr
        self.debug = debug

    def filter_results(self, results, min_confidence=0.6):
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
        # frames = [resize_percent(x, 50) for x in frames]
        device = 0 if sys.platform != "darwin" else "mps"
        processed = []
        for result, frame in zip(
            self.detection_model(frames, device=device, verbose=False), frames
        ):
            text_mask = np.zeros_like(frame, dtype=frame.dtype)

            segmentation_results = self.segmentation_model(
                frame, device=device, verbose=False
            )[0]

            if segmentation_results.masks is not None:
                for seg in list(
                    map(lambda a: a.astype("int"), segmentation_results.masks.xy)
                ):
                    cv2.fillPoly(text_mask, [seg], (255, 255, 255))

            filtered = self.filter_results(result)

            frame_clean = inpaint_optimized(
                frame,
                text_mask,
                filtered,  # segmentation_results.boxes.xyxy.cpu().numpy()
            )
            # debug_image(frame_clean, "Cleaned")
            # debug_image(frame, "FRAME")
            # debug_image(text_mask, "text mask")
            # inpainted = inpaint(frame, cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY), 3)
            # debug_image(inpainted, "Inpainted")
            to_translate = []
            # First pass, mask all bubbles
            for bbox, cls, conf in filtered:
                # if conf < 0.65:
                #     continue

                # print(get_ocr(get_box_section(frame, box)))
                color = (0, 0, 255) if cls == 1 else (0, 255, 0)

                (x1, y1, x2, y2) = bbox

                class_name = result.names[cls]
                if class_name == "text_bubble":
                    bubble = frame[y1:y2, x1:x2]
                    bubble_clean = frame_clean[y1:y2, x1:x2]
                    bubble_text_mask = text_mask[y1:y2, x1:x2]
                    # debug_image(frame[y1:y2, x1:x2], "FRAME")
                    # debug_image(text_mask[y1:y2, x1:x2], "text mask")
                    # inpainted = inpaint(
                    #     frame[y1:y2, x1:x2],
                    #     cv2.cvtColor(text_mask[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY),
                    #     3,
                    # )
                    # debug_image(inpainted, "Inpainted")
                    # # if len(bubble.shape) < 3:
                    # #     continue
                    text_only, bubble_mask = generate_bubble_mask(
                        bubble, bubble_text_mask, bubble_clean
                    )

                    frame[y1:y2, x1:x2] = bubble_clean
                    text_bounds = get_bounds_for_text(bubble_mask)
                    to_translate.append([bbox, bubble, text_only, text_bounds])

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

            # second pass, fix intersecting text areas
            for i in range(len(to_translate)):
                bbox_a = to_translate[i][0]
                text_bounds_a_local = to_translate[i][3]
                text_bounds_a = [
                    [
                        text_bounds_a_local[0][0] + bbox_a[0],
                        text_bounds_a_local[0][1] + bbox_a[1],
                    ],
                    [
                        text_bounds_a_local[1][0] + bbox_a[0],
                        text_bounds_a_local[1][1] + bbox_a[1],
                    ],
                ]
                for x in range(len(to_translate)):
                    if x == i:
                        continue

                    bbox_b = to_translate[x][0]
                    text_bounds_b_local = to_translate[x][3]
                    text_bounds_b = [
                        [
                            text_bounds_b_local[0][0] + bbox_b[0],
                            text_bounds_b_local[0][1] + bbox_b[1],
                        ],
                        [
                            text_bounds_b_local[1][0] + bbox_b[0],
                            text_bounds_b_local[1][1] + bbox_b[1],
                        ],
                    ]

                    fix_result = fix_intersection(
                        text_bounds_a[0],
                        text_bounds_a[1],
                        text_bounds_b[0],
                        text_bounds_b[1],
                    )
                    found_intersection = fix_result[4]
                    if found_intersection:
                        to_translate[i][3] = [
                            [
                                fix_result[0][0] - bbox_a[0],
                                fix_result[0][1] - bbox_a[1],
                            ],
                            [
                                fix_result[1][0] - bbox_a[0],
                                fix_result[1][1] - bbox_a[1],
                            ],
                        ]
                        to_translate[x][3] = [
                            [
                                fix_result[2][0] - bbox_b[0],
                                fix_result[2][1] - bbox_b[1],
                            ],
                            [
                                fix_result[3][0] - bbox_b[0],
                                fix_result[3][1] - bbox_b[1],
                            ],
                        ]
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
            for bbox, bubble, text_as_image, text_bounds in to_translate:
                (x1, y1, x2, y2) = bbox

                if self.translator and self.ocr:
                    translation = self.translator(self.ocr, text_as_image)
                    frame[y1:y2, x1:x2] = draw_text_in_bubble(
                        bubble, text_bounds, translation
                    )
                else:
                    frame[y1:y2, x1:x2] = draw_text_in_bubble(bubble, text_bounds)

            processed.append(frame)
        return processed
