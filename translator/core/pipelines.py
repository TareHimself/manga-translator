import cv2
import numpy as np
import sys
from ultralytics import YOLO
from requests.utils import requote_uri
from translator.utils import (
    get_outline_color,
    mask_text_and_make_bubble_mask,
    draw_text_in_bubble,
    get_bounds_for_text,
    TranslatorGlobals,
    inpaint_optimized,
    display_image,
    transform_sample,
    adjust_contrast_brightness,
    get_average_color,
    luminance_similarity,
    has_white
)
import traceback
import threading
import torch
import concurrent
from translator.color_detect.models import get_color_detection_model
from translator.core.translators import Translator
from translator.core.ocr import BaseOcr


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
        color_detect_model="models/color_detection.pt",
        translator=Translator(),
        ocr=BaseOcr(),
        font_file="fonts/animeace2_reg.ttf",
        debug=False,
    ) -> None:
        self.segmentation_model = YOLO(seg_model)
        self.detection_model = YOLO(detect_model)
        try:
            self.color_detect_model = get_color_detection_model(weights_path=color_detect_model,device=torch.device("cuda:0"))
            self.color_detect_model.eval()
        except:
            self.color_detect_model = None
            traceback.print_exc()

        self.translator = translator
        self.ocr = ocr
        self.debug = debug
        self.font_file = font_file
        self.frame_process_mutex = threading.Lock()

    def filter_results(self, results, min_confidence=0.1):
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
                filtered.append((box, results.names[obj_class], conf))

        return filtered
    

    def process_ml_results(self,detect_result,seg_result,frame):
        text_mask = np.zeros_like(frame, dtype=frame.dtype)

        if seg_result.masks is not None: # Fill in segmentation results
            for seg in list(
                map(lambda a: a.astype("int"), seg_result.masks.xy)
            ):
                cv2.fillPoly(text_mask, [seg], (255, 255, 255))
            

        detect_result = self.filter_results(detect_result)
        
        for bbox, cls, conf in detect_result: # fill in text free results
                if cls == "text_free":
                    (x1, y1, x2, y2) = bbox
                    text_mask = cv2.rectangle(text_mask,(x1,y1),(x2,y2),(255,255,255),-1)

        frame_clean,text_mask = inpaint_optimized(
            frame,
            text_mask,
            detect_result,  # segmentation_results.boxes.xyxy.cpu().numpy()
        )
        
        return frame,frame_clean,text_mask,detect_result

    def process_frame(self,detect_result,seg_result,frame):
        frame,frame_clean,text_mask,detect_result = self.process_ml_results(detect_result,seg_result,frame)


        to_translate = []
        # First pass, mask all bubbles
        for bbox, cls, conf in detect_result:
            # if conf < 0.65:
            #     continue

            # print(get_ocr(get_box_section(frame, box)))
            color = (0, 0, 255) if cls == 1 else (0, 255, 0)

            (x1, y1, x2, y2) = bbox

            class_name = cls
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
                if has_white(bubble_text_mask):
                    text_only, bubble_mask = mask_text_and_make_bubble_mask(
                        bubble, bubble_text_mask, bubble_clean
                    )

                    frame[y1:y2, x1:x2] = bubble_clean
                    text_draw_bounds = get_bounds_for_text(bubble_mask)
                    to_translate.append([bbox, bubble, text_only, text_draw_bounds])
                    # debug_image(text_only,"Text Only")
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
        text_colors = [TranslatorGlobals.COLOR_BLACK for x in to_translate]
        
        if self.color_detect_model is not None:
            with torch.no_grad(): # model needs work
                with torch.inference_mode():
                    with self.frame_process_mutex: # this may not be needed
                        def fix_image(frame):
                            # frame = adjust_contrast_brightness(frame,contrast=2)
                            # cv2.GaussianBlur(, (size_dil, size_dil), 0)
                            # final_mask_dilation = 6
                            # kernel = np.ones((final_mask_dilation,final_mask_dilation),np.uint8)
                            # return cv2.dilate(frame,kernel,iterations = 1)
                            return frame
                        
                        images = [fix_image(x[2].copy()) for x in to_translate]
                        # images = [x[2].copy() for x in to_translate]
                        # [debug_image(x,"To Detect") for x in images]
                        text_colors = [(x.cpu().numpy() * 255).astype(np.uint8) for x in self.color_detect_model(torch.stack([transform_sample(y) for y in images]).to(torch.device("cuda:0")))]
        else:
            print("Using black since color detect model is not valid")
            
        for i in range(len(to_translate)):
            bbox, bubble, text_as_image, text_draw_bounds = to_translate[i]
            text_color = text_colors[i]
            (x1, y1, x2, y2) = bbox

            if self.translator and self.ocr:
                # debug_image(text_as_image,"Item")
                translation = self.translator(self.ocr, text_as_image)
                if len(translation.strip()):
                    draw_surface = frame[y1:y2, x1:x2]
                    # draw_surface_color = get_average_color(draw_surface)

                    # if luminance_similarity(np.array(text_color),TranslatorGlobals.COLOR_BLACK) >= .7:
                    #     text_color = TranslatorGlobals.COLOR_BLACK

                    # elif luminance_similarity(np.array(text_color),TranslatorGlobals.COLOR_WHITE) >= .9:
                    #     text_color = TranslatorGlobals.COLOR_WHITE

                    outline_color = get_outline_color(draw_surface,text_color)
                    
                    # if outline_color is None:

                    #     print("Similarities",luminance_similarity(text_color,draw_surface_color))

                    frame[y1:y2, x1:x2] = draw_text_in_bubble(
                        bubble, text_draw_bounds, translation,font_file=self.font_file,color=text_color, outline_color=outline_color
                    )
            

        
        return frame
    
    def __call__(self, images: list[np.ndarray],yolo_device = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "mps" if sys.platform != "darwin" else torch.device('cpu')) -> list[np.ndarray]:
        # frames = [resize_percent(x, 50) for x in frames]
        to_process = [x for x in zip(
            self.detection_model(images, device=yolo_device, verbose=False),self.segmentation_model(
                images, device=yolo_device, verbose=False
            ), images
        )]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_process)) as executor:
            futures = []
            for i in range(len(to_process)):
                detect_result,seg_result, frame = to_process[i]
                futures.append(executor.submit(self.process_frame,detect_result=detect_result,seg_result=seg_result,frame=frame))

            
            return list(map(lambda a: a.result(),futures))
