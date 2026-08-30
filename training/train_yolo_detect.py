from ultralytics import YOLO, settings
import torch
import os
import simple_parsing
from dataclasses import dataclass


@dataclass
class Config:
    dataset: str  # Path to the dataset 'data.yaml'
    output: str = os.path.join(".", "trained_detect.pt")  # Trained model output path
    checkpoint: str = "yolo11n.pt"  # The checkpoint to start training from
    device: str = "cuda:0"  # pytorch device to train with
    patience: int = 5
    project: str = "Manga Translator Detection"


if __name__ == "__main__":
    config: Config = simple_parsing.parse(Config)
    device = torch.device(config.device)

    run_root = os.path.abspath(os.path.join(os.getcwd(), "runs"))
    os.makedirs(run_root, exist_ok=True)
    settings.update({"runs_dir": run_root, "wandb": True})

    project_dir = os.path.abspath(os.path.join(run_root, "detect", config.project))

    # Load a pretrained YOLO11n model
    model = YOLO(model=config.checkpoint, task="detect")

    results = model.train(
        data=config.dataset,
        patience=config.patience,
        imgsz=640,
        batch=0.9,
        device=device,
        epochs=1000,
        project=config.project,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=0.0,#5.0,
        translate=0.0,#0.1,
        scale=0.0,#0.5,
        fliplr=0.5,
        mosaic=0.0,#1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=30,
        rect=True
    )

    model.save(config.output)
