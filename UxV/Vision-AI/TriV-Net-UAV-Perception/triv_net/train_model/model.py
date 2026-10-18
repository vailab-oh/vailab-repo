import torch
import torch.nn as nn
from .utils import Anchors, BBoxTransform, ClipBoxes
from .encoder import Encoder_Resnet
from .decoder import ClassificationModel, FPN, Regressor


class models(nn.Module):
    def __init__(self, cfg):
        super(models, self).__init__()

        self.cfg = cfg
        camC = 128

        self.camencode = Encoder_Resnet(
            pretrained=bool(cfg["model"].get("pretrained", True))
        )
        self.decoder = FPN([2048, 1024, 512], feat_c=camC)

        if cfg["model_conf"]["od"]:
            num_classes = int(cfg.get("od_num_classes", cfg.get("class_id", 5)))

            aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
            num_scales = len([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

            num_anchors = len(aspect_ratios) * num_scales
            self.regressor = Regressor(
                in_channels=camC,
                num_anchors=num_anchors,
                num_layers=3,
                pyramid_levels=5,
            )

            self.classifier = ClassificationModel(
                num_features_in=128, num_classes=num_classes
            )

            self.anchors = Anchors(anchor_scale=1.25)
            self.regressBoxes = BBoxTransform()
            self.clipBoxes = ClipBoxes()

        if cfg["model_conf"]["Seg"]:
            seg_num_classes = int(cfg.get("seg_num_classes", 106))
            self.seg_output = nn.Sequential(
                nn.Conv2d(camC, 64, kernel_size=3, padding=1, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, seg_num_classes, kernel_size=1, padding=0),
            )

        if cfg["model_conf"]["Depth"]:
            self.depth_output = nn.Sequential(
                nn.Conv2d(camC, 64, kernel_size=3, padding=1, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0),
            )

    def forward(self, x):
        feats = self.camencode(x)
        out_x = self.decoder(feats)

        if self.cfg["model_conf"]["Seg"]:
            seg = self.seg_output(out_x[0])
        else:
            seg = None

        if self.cfg["model_conf"]["Depth"]:
            depth = torch.sigmoid(self.depth_output(out_x[0])) * 80.0
        else:
            depth = None

        if self.cfg["model_conf"]["od"]:
            regression = self.regressor(out_x)
            classification = torch.cat(
                [self.classifier(feature) for feature in out_x], dim=1
            )
        else:
            classification = None
            regression = None

        return seg, depth, classification, regression
