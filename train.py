from ultralytics import YOLO

# detection
# model = YOLO("yolov8n.pt")  # YOLO("yolov8m-seg.pt")

# model.train(data="train_detect.yaml", epochs=200, batch=30, device=0, workers=0)

# #segmentation
# model = YOLO(
#     "D:\\Github\\manga-translator\\runs\\segment\\train12\\weights\\best.pt"
# )  # YOLO("yolov8m-seg.pt")

# model.train(
#     data="train_seg.yaml", epochs=200, batch=30, patience=100, device=0, workers=0
# )
