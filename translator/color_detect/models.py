from typing import Callable
import timm
import torch
import torch.nn as nn


def get_color_detection_model(
    build_final_layer: Callable[[int], nn.Module] = lambda a: nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(a, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 3),
        nn.Sigmoid(),
    ),
    channels=3,
    model_name="efficientnet_b4",
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
