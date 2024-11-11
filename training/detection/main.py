from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("yolov10l")
    model.train(data="config.yaml",batch=0.8,patience = 30,device=0,verbose=True,imgsz=640)
