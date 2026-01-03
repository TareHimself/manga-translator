
from ultralytics import YOLO
import torch
import os
import simple_parsing
from dataclasses import dataclass

@dataclass
class Config:
    output: str = os.path.join(".","trained_detect.pt") # Trained model output path
    dataset: str # Path to the dataset 'data.yaml'
    checkpoint: str = './yolo11m-detect.pt' # The checkpoint to start training from
    device: str = 'cuda:0' # pytorch device to train with
    
if __name__ == '__main__':
    config: Config = simple_parsing.parse_known_args(Config)
    device = torch.device(config.device)
    # Load a pretrained YOLO11n model
    model = YOLO(model=config.checkpoint,task="detect")

    results = model.train(data=config.dataset,patience=3, imgsz=640,batch=0.8,device=device,epochs=20000)
    
    model.save(config.output)