from typing import Callable
import timm
import torch
import torch.functional as F
import torch.nn as nn

class ClampModule(nn.Module):

    def __init__(self,min = 0, max = 1) -> None:
        super().__init__()
        self.min = min
        self.max = max

    
    def forward(self,data):
        return torch.clamp(data,min=self.min,max=self.max)
    
def get_color_detection_model(
    build_final_layer: Callable[[int], nn.Module] = lambda a: nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(a, 1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 6),
        ClampModule(min=0,max=1)
    ),
    channels=3,
    model_name="resnet18",
    pretrained=False,
    device=torch.device("cpu"),
    weights_path=None,
):
    model = timm.create_model(
        model_name=model_name, pretrained=pretrained, in_chans=channels, num_classes=3
    )
    # model = timm.create_model("resnet18",pretrained=True,in_chans=3,num_classes=3)
    classifier = model.default_cfg["classifier"]

    in_features = getattr(
        model, classifier
    ).in_features  # Get the number of input features for the final layer

    setattr(
        model, classifier, build_final_layer(in_features)
    )  # Replace the final layer

    

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    model.to(device)

    return model
