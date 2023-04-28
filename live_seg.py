from mss import mss
from manga_ocr import MangaOcr
import cv2
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image
import requests
from requests.utils import requote_uri

model = YOLO("seg.pt")
manga_ocr = MangaOcr()


def translate(texts):
    data = [("target_lang", "EN-US"), ("source_lang", "JA")]
    for text in texts:
        data.append(("text", text))
    uri = f"https://api-free.deepl.com/v2/translate?{'&'.join([f'{data[i][0]}={data[i][1]}' for i in range(len(data))])}"
    print(uri)
    uri = requote_uri(uri)
    print(uri)
    return requests.post(
        uri,
        headers={
            "Authorization": f"DeepL-Auth-Key 0e0edc82-62cd-d0d5-422b-004d07e029ef:fx",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    ).json()["translations"]


def get_ocr(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return manga_ocr(Image.fromarray(frame))


def filter_results(frame, results, min_confidence=0.65):
    if len(results) < 1:
        return None

    height, width, channels = frame.shape
    bounding_boxes = np.array(results.boxes.xyxy.cpu(), dtype="int")

    classes = np.array(results.boxes.cls.cpu(), dtype="int")

    confidence = np.array(results.boxes.conf.cpu(), dtype="float")

    return zip(
        bounding_boxes,
        classes,
        list(map(lambda a: a.astype("int"), results.masks.xy)),
        confidence,
    )


def get_box_section(frame, box):
    (x1, y1, x2, y2) = box

    return frame[y1:y2, x1:x2]


def test_frame(frame):
    results = model.predict(frame, device=0)[0]

    filtered = filter_results(frame, results)
    if filtered is not None:
        for box, cls, seg, conf in filtered:
            # if conf < 0.65:
            #     continue

            # print(get_ocr(get_box_section(frame, box)))
            color = (0, 0, 255) if cls == 1 else (0, 255, 0)

            (x, y, x2, y2) = box

            cv2.putText(
                frame,
                str(f"{results.names[cls]} | {conf * 100:.1f}%"),
                (x, y - 20),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color,
                2,
            )
            # print(seg)
            cv2.fillPoly(frame, [seg], (255, 255, 255))
            # cv2.polylines(frame, [seg], True, color, 2)
            # cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.waitKey(1)


# print("Translation of 俺らまだ恋人じゃないし", translate(["俺らまだ恋人じゃないし", "俺らまだ恋人じゃないし"]))
with mss() as sct:
    monitor = sct.monitors[1]
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB), (int(1920 / 2), int(1080 / 2))
        )

        test_frame(frame)


# for image in os.listdir("test"):
#     if not image.endswith(".png"):
#         continue

#     frame = cv2.imread(os.path.join("test", image))
#     test_frame(frame)
