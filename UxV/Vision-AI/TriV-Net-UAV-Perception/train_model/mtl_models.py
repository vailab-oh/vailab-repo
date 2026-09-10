import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as tv_models
import math


def conv_bn_act(in_c, out_c, k=3, s=1, p=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True),
    )


class TorchvisionConvNeXtTiny(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = tv_models.convnext_tiny(weights=weights).features
        self.channels = [192, 384, 768]

    def forward(self, x):
        outs = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in (3, 5, 7):
                outs.append(x)
        return outs


class TimmBackbone(nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
            out_indices=(2, 3, 4),
        )
        self.channels = self.model.feature_info.channels()

    def forward(self, x):
        return self.model(x)


def build_backbone(name, pretrained=False):
    key = name.lower().replace("_", "-")
    if key in ("convnext-tiny", "convnexttiny"):
        return TorchvisionConvNeXtTiny(pretrained=pretrained)
    if key in ("efficientnetv2-s", "efficientnet-v2-s", "tf-efficientnetv2-s"):
        return TimmBackbone("tf_efficientnetv2_s", pretrained=pretrained)
    if key in ("mobilenetv3-l", "mobilenetv3-large", "mobilenetv3-large-100"):
        return TimmBackbone("mobilenetv3_large_100", pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {name}")


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lateral3 = conv_bn_act(c3, out_channels, 1, 1, 0)
        self.lateral4 = conv_bn_act(c4, out_channels, 1, 1, 0)
        self.lateral5 = conv_bn_act(c5, out_channels, 1, 1, 0)
        self.out3 = conv_bn_act(out_channels, out_channels)
        self.out4 = conv_bn_act(out_channels, out_channels)
        self.out5 = conv_bn_act(out_channels, out_channels)
        self.p6 = conv_bn_act(out_channels, out_channels, 3, 2, 1)
        self.p7 = conv_bn_act(out_channels, out_channels, 3, 2, 1)

    def forward(self, feats):
        c3, c4, c5 = feats[-3:]
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.out3(p3)
        p4 = self.out4(p4)
        p5 = self.out5(p5)
        p6 = self.p6(p5)
        p7 = self.p7(p6)
        return [p3, p4, p5, p6, p7]


class LRASPPHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.low = conv_bn_act(channels, channels, 1, 1, 0)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            conv_bn_act(channels, channels),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, pyramid, out_size):
        x = self.low(pyramid[0])
        x = x * self.context(x)
        x = self.project(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, channels, out_channels=128, rates=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                self.blocks.append(conv_bn_act(channels, out_channels, 1, 1, 0))
            else:
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                    )
                )
        self.project = conv_bn_act(out_channels * len(rates), out_channels, 1, 1, 0)

    def forward(self, x):
        return self.project(torch.cat([block(x) for block in self.blocks], dim=1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.aspp = ASPP(channels, channels)
        self.low = conv_bn_act(channels, 48, 1, 1, 0)
        self.fuse = nn.Sequential(
            conv_bn_act(channels + 48, channels),
            conv_bn_act(channels, channels),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, pyramid, out_size):
        low = self.low(pyramid[0])
        high = F.interpolate(self.aspp(pyramid[2]), size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([low, high], dim=1))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class FPNDepthDecoder(nn.Module):
    def __init__(self, channels, multi_scale=False):
        super().__init__()
        self.multi_scale = multi_scale
        in_c = channels * 3 if multi_scale else channels
        self.head = nn.Sequential(
            conv_bn_act(in_c, channels),
            conv_bn_act(channels, channels),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, pyramid, out_size):
        if self.multi_scale:
            base_size = pyramid[0].shape[-2:]
            x = torch.cat(
                [
                    pyramid[0],
                    F.interpolate(pyramid[1], size=base_size, mode="bilinear", align_corners=False),
                    F.interpolate(pyramid[2], size=base_size, mode="bilinear", align_corners=False),
                ],
                dim=1,
            )
        else:
            x = pyramid[0]
        depth = torch.sigmoid(self.head(x)) * 80.0
        return F.interpolate(depth, size=out_size, mode="bilinear", align_corners=False)


class ODHead(nn.Module):
    def __init__(self, channels, num_classes, head_type="yolov8"):
        super().__init__()
        self.head_type = head_type.lower()
        self.num_classes = num_classes
        self.strides = [8, 16, 32, 64, 128]
        self.register_buffer("_box_encoding_version", torch.tensor(2, dtype=torch.uint8))
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        for _ in range(5):
            self.cls_heads.append(self._tower(channels, num_classes))
            self.box_heads.append(self._tower(channels, 4))
        self._init_detection_biases()

    def _init_detection_biases(self, prior_prob=0.01):
        cls_bias = -math.log((1.0 - prior_prob) / prior_prob)
        for head in self.cls_heads:
            for module in reversed(head):
                if isinstance(module, nn.Conv2d):
                    nn.init.constant_(module.bias, cls_bias)
                    break
        for head in self.box_heads:
            for module in reversed(head):
                if isinstance(module, nn.Conv2d):
                    nn.init.constant_(module.bias, 1.0)
                    break

    def _tower(self, channels, out_channels):
        if self.head_type == "rtmdet":
            block = lambda: conv_bn_act(channels, channels, 3, 1, 1, groups=channels)
            return nn.Sequential(block(), conv_bn_act(channels, channels, 1, 1, 0), nn.Conv2d(channels, out_channels, 1))
        if self.head_type == "fcos":
            return nn.Sequential(conv_bn_act(channels, channels), conv_bn_act(channels, channels), nn.Conv2d(channels, out_channels, 3, padding=1))
        return nn.Sequential(
            conv_bn_act(channels, channels),
            conv_bn_act(channels, channels),
            nn.Conv2d(channels, out_channels, 1),
        )

    def forward(self, pyramid):
        cls_logits = []
        box_ltrb = []
        for feat, cls_head, box_head in zip(pyramid, self.cls_heads, self.box_heads):
            cls_logits.append(cls_head(feat))
            # Small boxes can require signed offsets when the cell center lies outside the box.
            box_ltrb.append(box_head(feat))
        return {"cls_logits": cls_logits, "box_ltrb": box_ltrb, "strides": self.strides}


class models(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        pretrained = bool(model_cfg.get("pretrained", True))
        fpn_channels = int(model_cfg.get("fpn_channels", 128))
        self.backbone = build_backbone(model_cfg.get("backbone", model_cfg.get("encoder", "mobilenetv3-l")), pretrained=pretrained)
        self.neck = FPN(self.backbone.channels, out_channels=fpn_channels)

        conf = cfg.get("model_conf", {})
        if conf.get("Seg", False):
            seg_classes = int(cfg.get("seg_num_classes", 12))
            seg_head = model_cfg.get("seg_head", "lraspp").lower()
            if seg_head in ("deeplabv3plus", "deeplabv3+"):
                self.seg_head = DeepLabV3PlusHead(fpn_channels, seg_classes)
            else:
                self.seg_head = LRASPPHead(fpn_channels, seg_classes)
        else:
            self.seg_head = None

        if conf.get("Depth", False):
            depth_head = model_cfg.get("depth_head", "fpn_depth").lower()
            self.depth_head = FPNDepthDecoder(fpn_channels, multi_scale=depth_head in ("fpn_multiscale", "fpn_multi_scale"))
        else:
            self.depth_head = None

        if conf.get("od", False):
            self.od_head = ODHead(fpn_channels, int(cfg.get("od_num_classes", 5)), model_cfg.get("od_head", "yolov8"))
        else:
            self.od_head = None

    def forward(self, x):
        out_size = x.shape[-2:]
        feats = self.backbone(x)
        pyramid = self.neck(feats)
        seg = self.seg_head(pyramid, out_size) if self.seg_head is not None else None
        depth = self.depth_head(pyramid, out_size) if self.depth_head is not None else None
        od = self.od_head(pyramid) if self.od_head is not None else None
        return seg, depth, od, None
