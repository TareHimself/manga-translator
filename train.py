from ultralytics import YOLO

# detection
# model = YOLO("yolov8m.pt")  # YOLO("yolov8m-seg.pt")

# model.train(
#     data="train_detect.yaml", epochs=400, batch=-1, device=0, workers=0, patience=100
# )

# #segmentation
model = YOLO("yolov8m-seg.pt")  # YOLO("yolov8m-seg.pt")

model.train(
    data="train_seg.yaml", epochs=400, batch=10, patience=100, device=0, workers=0
)
