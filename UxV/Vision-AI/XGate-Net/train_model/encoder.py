import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models as tv

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Encoder_EfficientNet(nn.Module):
    def __init__(self, C):
        super(Encoder_EfficientNet, self).__init__()
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

    def forward(self, x):
        endpoints = dict()

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        return [endpoints['reduction_5'],endpoints['reduction_4'],endpoints['reduction_3']]
    

class Encoder_Resnet(nn.Module):
    def __init__(self, C):
        super(Encoder_Resnet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.feat_names = [ 'layer1', 'layer2', 'layer3', 'layer4']
        self.feat_out_channels = [256, 512, 1024, 2048]



    def forward(self, x):
        features = [x]
        skip_feat = []
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                if feature.size(1) in self.feat_out_channels:
                    skip_feat.append(feature)


        return [skip_feat[3],skip_feat[2],skip_feat[1]]
    

class Encoder_DenseNet121(nn.Module):
    def __init__(self, C=None, pretrained: bool = True):
        super().__init__()
        try:
            weights = tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            d = tv.densenet121(weights=weights).features
        except Exception:
            d = tv.densenet121(pretrained=pretrained).features

        self.conv0       = d.conv0
        self.norm0       = d.norm0
        self.relu0       = d.relu0
        self.pool0       = d.pool0
        self.denseblock1 = d.denseblock1
        self.transition1 = d.transition1
        self.denseblock2 = d.denseblock2
        self.transition2 = d.transition2
        self.denseblock3 = d.denseblock3
        self.transition3 = d.transition3
        self.denseblock4 = d.denseblock4
        self.norm5       = d.norm5

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x); x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x)
        x = self.transition1(x)

        c3 = self.denseblock2(x)
        x  = self.transition2(c3)
        c4 = self.denseblock3(x)
        x  = self.transition3(c4)
        c5 = self.denseblock4(x)
        c5 = self.norm5(c5)

        return [c5, c4, c3]
