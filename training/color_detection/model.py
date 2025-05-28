from typing import Callable
import timm
import torch
import torch.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from vit_pytorch import ViT
from constants import IMAGE_SIZE
train_augmenter = transforms.Compose(
    [
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.RandomAutocontrast(),
        # transforms.RandomRotation(degrees=(0, 360),interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip()
    ]
)

def get_timm_model(
    build_final_layer: Callable[[int], nn.Module] = lambda a: nn.Linear(a, 2),
    channels=3,
    model_name="resnet18",
    pretrained=False,
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

    return model

class ClampModule(nn.Module):

    def __init__(self,min = 0, max = 1) -> None:
        super().__init__()
        self.min = min
        self.max = max

    
    def forward(self,data):
        return torch.clamp(data,min=self.min,max=self.max)

class ResnetFeatureExtractor(nn.Module):
    def __init__(self,out_features=1024) -> None:
        super().__init__()
        self.ex = get_timm_model(lambda x : nn.Linear(x, out_features),
            channels=3,
            model_name="resnet18")

    def forward(self,x):
        return self.ex(x)
    
class ViTFeatureExtractor(nn.Module):
    def __init__(self,out_features=1024,image_size=IMAGE_SIZE) -> None:
        super().__init__()
        self.ex = ViT(
            image_size = image_size,
            patch_size = 32,
            num_classes = out_features,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.3,
            emb_dropout = 0.1
        )

    def forward(self,x):
        return self.ex(x)
        
class ColorDetectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ex = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.ex.fc = nn.Identity()
        self.fc_color = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 3 for text color, 3 for border color
        )
        self.fc_border = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),   # 1 for binary
            nn.Sigmoid()
        )

    def forward(self,x):
        # if self.training:
        #     x = train_augmenter(x)
        # to_display = apply_transforms_inverse(x[0] * 255)
        # display_image(to_display,"Train Augmentation Debug")
        features = self.ex(x)

        return self.fc_color(features), self.fc_border(features)
        
def get_color_detection_model(
    device=torch.device("cpu"),
    weights_path=None,
):
    model = ColorDetectionModel()

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    model.to(device)

    return model