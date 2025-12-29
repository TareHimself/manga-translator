
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = torch.device('cuda:0')
    # Load a pretrained YOLO11n model
    model = YOLO(model='yolo11n-seg.yaml',task="segment")

    results = model.train(data=r"",patience=30, imgsz=640,batch=-1,device=device,epochs=20000)
    
    model.save('./yolo11n-seg.pt')