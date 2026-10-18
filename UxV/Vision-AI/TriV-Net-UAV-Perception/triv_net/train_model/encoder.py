import torch.nn as nn
import torchvision.models as models


class Encoder_Resnet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = models.resnet50(weights=weights)
        self.feat_names = ("layer1", "layer2", "layer3", "layer4")

    def forward(self, x):
        features = []
        for name, layer in self.base_model.named_children():
            if name in ("avgpool", "fc"):
                continue
            x = layer(x)
            if name in self.feat_names:
                features.append(x)
        return [features[3], features[2], features[1]]
