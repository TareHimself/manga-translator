import timm
import torch
import torch.nn as nn

def get_color_detection_model(weights_path=None,device=torch.device('cpu')):
    model = timm.create_model("efficientnet_b4",pretrained=True,in_chans=3,num_classes=3)
    # model = timm.create_model("resnet18",pretrained=True,in_chans=3,num_classes=3)
    classifier = model.default_cfg['classifier']
    

    
    in_features = getattr(model,classifier).in_features  # Get the number of input features for the final layer

    setattr(model,classifier,nn.Sequential(nn.Dropout(.4),nn.Linear(in_features, 128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128, 3),nn.Sigmoid())) # Replace the final layer
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.to(device)
    return model