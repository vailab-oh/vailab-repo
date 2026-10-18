import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_c, out_c, k=1, s=1, p=0, act=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False), nn.BatchNorm2d(out_c)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class FPN(nn.Module):
    def __init__(self, in_channels, feat_c=256):
        super().__init__()
        self.input_proj3 = conv_bn_act(in_channels[2], feat_c, 1, 1, 0)
        self.input_proj4 = conv_bn_act(in_channels[1], feat_c, 1, 1, 0)
        self.input_proj5 = conv_bn_act(in_channels[0], feat_c, 1, 1, 0)

        self.smooth3 = conv_bn_act(feat_c, feat_c, 3, 1, 1)
        self.smooth4 = conv_bn_act(feat_c, feat_c, 3, 1, 1)
        self.smooth5 = conv_bn_act(feat_c, feat_c, 3, 1, 1)
        self.p6 = conv_bn_act(feat_c, feat_c, 3, 2, 1)
        self.p7 = conv_bn_act(feat_c, feat_c, 3, 2, 1)

    def forward(self, feats):
        x5, x4, x3 = feats[-3:]

        p3 = self.input_proj3(x3)
        p4 = self.input_proj4(x4)
        p5 = self.input_proj5(x5)

        p5_out = self.smooth5(p5)
        p4_out = self.smooth4(
            p4 + F.interpolate(p5_out, scale_factor=2, mode="nearest")
        )
        p3_out = self.smooth3(
            p3 + F.interpolate(p4_out, scale_factor=2, mode="nearest")
        )
        p6_out = self.p6(p5_out)
        p7_out = self.p7(F.relu(p6_out))
        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class Conv2dStaticSamePadding(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        groups=1,
        dilation=1,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            groups=groups,
        )
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (
            (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
            - w
            + self.kernel_size[1]
        )
        extra_v = (
            (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
            - h
            + self.kernel_size[0]
        )

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1
        )

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [
                SeparableConvBlock(
                    in_channels, in_channels, norm=False, activation=False
                )
                for i in range(num_layers)
            ]
        )
        self.bn_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
                        for i in range(num_layers)
                    ]
                )
                for j in range(pyramid_levels)
            ]
        )
        self.header = SeparableConvBlock(
            in_channels, num_anchors * 4, norm=False, activation=False
        )
        self.swish = Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchors=9,
        num_classes=80,
        prior=0.01,
        feature_size=128,
    ):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
