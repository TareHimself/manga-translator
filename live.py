from mss import mss
import cv2
from ultralytics import YOLO
import numpy as np
from pipelines import FullConversion


converter = FullConversion()
with mss() as sct:
    monitor = sct.monitors[1]
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB), (int(1920 / 1), int(1080 / 1))
        )

        frame = converter([frame])[0]

        if frame is not None:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)


# for image in os.listdir("test"):
#     if not image.endswith(".png"):
#         continue

#     frame = cv2.imread(os.path.join("test", image))
#     test_frame(frame)
