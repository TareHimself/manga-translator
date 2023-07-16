from ultralytics import YOLO

# detection
model = YOLO("./runs/detect/train9/weights/best.pt")  # YOLO("yolov8m-seg.pt")

model.train(
    data="train_detect.yaml", epochs=10000, batch=-1, device=0, workers=0, patience=100
)

# #segmentation
# model = YOLO("./models/segmentation.pt")  # YOLO("yolov8m-seg.pt")

# model.train(
#     data="train_seg.yaml", epochs=1000, batch=10, patience=100, device=0, workers=0
# )
