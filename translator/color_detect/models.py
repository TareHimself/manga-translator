import timm
import torch
import torch.nn as nn

def get_color_detection_model(weights_path=None,device=torch.device('cpu')):
    model = timm.create_model("resnet18",pretrained=True,in_chans=3,num_classes=3)
    in_features = model.fc.in_features  # Get the number of input features for the final layer
    model.fc = nn.Sequential(nn.Dropout(.4),nn.Linear(in_features, 128),nn.ReLU(1),nn.Dropout(.4),nn.Linear(128, 3),nn.ReLU(1))  # Replace the final layer
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.to(device)
    return model