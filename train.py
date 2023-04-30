from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")  # YOLO("yolov8m-seg.pt")

model.train(data="train_seg.yaml", epochs=70, batch=10, device=0, workers=0)
