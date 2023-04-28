from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

model.train(data="seg.yaml", epochs=100, batch=10, device=0, workers=0)
