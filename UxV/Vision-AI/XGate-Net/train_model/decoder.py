import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torchvision.models.resnet import resnet18 
from torchvision.models import densenet121
import math

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

class Bev_Decoder_EfficientNet(nn.Module):
    def __init__(self, inC):
        super(Bev_Decoder_EfficientNet, self).__init__()

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0",pretrained=False)
        self.trunk._conv_stem = nn.Conv2d(inC, 32, kernel_size=3, padding=1, bias=False)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(112, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(320, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)


        self.up1_BEV = Up(64 + 64, 64, scale_factor=2)
        self.up2_BEV = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_vehicle = Up(64 + 64, 64, scale_factor=2)
        self.up2_vehicle = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_Lane = Up(64 + 64, 64, scale_factor=2)
        self.up2_Lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_walker = Up(64 + 64, 64, scale_factor=2)
        self.up2_walker = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )


    def get_eff_feature(self, x):
        endpoints = []

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) 
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints.append(prev_x)
            prev_x = x

        endpoints.append(x)
        del endpoints[0]
        return endpoints

    def forward(self, x):
        pyramid = self.get_eff_feature(x)

        ups=self.up1(pyramid[2])
        ups2=self.up2(pyramid[3])

        DAS = self.up1_BEV(ups2, ups)
        DAS = self.up2_BEV(DAS)

        Vehicle = self.up1_vehicle(ups2, ups)
        Vehicle = self.up2_vehicle(Vehicle)

        Lane = self.up1_Lane(ups2, ups)
        Lane = self.up2_Lane(Lane)
        
        walker = self.up1_walker(ups2, ups)
        walker = self.up2_walker(walker)

        return DAS,Lane,Vehicle,walker
    
class Bev_decoder_Resnet(nn.Module):
    def __init__(self, inC):
        super(Bev_decoder_Resnet, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1_BEV = Up(64 + 256, 64, scale_factor=2)
        self.up2_BEV = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_vehicle = Up(64 + 256, 64, scale_factor=2)
        self.up2_vehicle = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_Lane = Up(64 + 256, 64, scale_factor=2)
        self.up2_Lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_walker = Up(64 + 256, 64, scale_factor=2)
        self.up2_walker = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        BEV_SEG = self.up1_BEV(x2, x1)
        BEV_SEG = self.up2_BEV(BEV_SEG)

        DAS = self.up1_BEV(x2, x1)
        DAS = self.up2_BEV(DAS)

        Vehicle = self.up1_vehicle(x2, x1)
        Vehicle = self.up2_vehicle(Vehicle)

        Lane = self.up1_Lane(x2, x1)
        Lane = self.up2_Lane(Lane)
        
        walker = self.up1_walker(x2, x1)
        walker = self.up2_walker(walker)

        return DAS,Lane,Vehicle,walker

class Bev_Decoder_DenseNet121(nn.Module):
    def __init__(self, inC):
        super().__init__()
        self.trunk = densenet121(pretrained=False).features
        self.trunk.conv0 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.up1_BEV = Up(64 + 64, 64, scale_factor=2)
        self.up2_BEV = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_vehicle = Up(64 + 64, 64, scale_factor=2)
        self.up2_vehicle = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_Lane = Up(64 + 64, 64, scale_factor=2)
        self.up2_Lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_walker = Up(64 + 64, 64, scale_factor=2)
        self.up2_walker = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def get_dn_feature(self, x):
        feats = []
        x = self.trunk.conv0(x)
        x = self.trunk.norm0(x)
        x = self.trunk.relu0(x)
        x = self.trunk.pool0(x)

        x = self.trunk.denseblock1(x)
        feats.append(x)
        x = self.trunk.transition1(x)

        x = self.trunk.denseblock2(x)
        feats.append(x)
        x = self.trunk.transition2(x)

        x = self.trunk.denseblock3(x)
        feats.append(x)
        x = self.trunk.transition3(x)

        x = self.trunk.denseblock4(x)
        x = self.trunk.norm5(x)
        x = F.relu(x, inplace=True)
        feats.append(x)

        return feats

    def forward(self, x):
        pyramid = self.get_dn_feature(x)
        ups  = self.up1(pyramid[2])
        ups2 = self.up2(pyramid[3])

        DAS     = self.up1_BEV(ups2, ups);     DAS     = self.up2_BEV(DAS)
        Vehicle = self.up1_vehicle(ups2, ups); Vehicle = self.up2_vehicle(Vehicle)
        Lane    = self.up1_Lane(ups2, ups);    Lane    = self.up2_Lane(Lane)
        walker  = self.up1_walker(ups2, ups);  walker  = self.up2_walker(walker)

        return DAS, Lane, Vehicle, walker



def conv_bn_act(in_c, out_c, k=1, s=1, p=0, act=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False),
              nn.BatchNorm2d(out_c)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)

class FastNormalizedFusion(nn.Module):

    def __init__(self, n_inputs):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32),
                              requires_grad=True)

    def forward(self, *xs):
        ws = F.relu(self.w)
        ws = ws / (ws.sum() + 1e-6)
        out = sum(w * x for w, x in zip(ws, xs) if x is not None)
        return out
class SeparableConv2d(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, k, 1, p, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class FPN(nn.Module):
    def __init__(self, in_channels,feat_c=256):
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
        p4_out = self.smooth4(p4 + F.interpolate(p5_out, scale_factor=2, mode='nearest'))
        p3_out = self.smooth3(p3 + F.interpolate(p4_out, scale_factor=2, mode='nearest'))
        p6_out = self.p6(p5_out)
        p7_out = self.p7(F.relu(p6_out))
        return [p3_out, p4_out, p5_out, p6_out, p7_out]
 
class BiFPN(nn.Module):
    def __init__(self, in_channels, feat_c=256):
        super().__init__()
        self.input_proj3 = conv_bn_act(in_channels[2], feat_c, 1, 1, 0)
        self.input_proj4 = conv_bn_act(in_channels[1], feat_c, 1, 1, 0)
        self.input_proj5 = conv_bn_act(in_channels[0], feat_c, 1, 1, 0)

        self.p6 = conv_bn_act(feat_c, feat_c, 3, 2, 1)
        self.p7 = conv_bn_act(feat_c, feat_c, 3, 2, 1)

        self.fuse_td4 = FastNormalizedFusion(2)
        self.fuse_td3 = FastNormalizedFusion(2)
        self.sepconv_td4 = SeparableConv2d(feat_c, feat_c)
        self.sepconv_td3 = SeparableConv2d(feat_c, feat_c)

        self.fuse_bu4 = FastNormalizedFusion(3)
        self.fuse_bu5 = FastNormalizedFusion(3)
        self.sepconv_bu4 = SeparableConv2d(feat_c, feat_c)
        self.sepconv_bu5 = SeparableConv2d(feat_c, feat_c)

    @staticmethod
    def _resize_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')

    def forward(self, feats):
        x5, x4, x3 = feats[-3:]

        p3 = self.input_proj3(x3)
        p4 = self.input_proj4(x4)
        p5 = self.input_proj5(x5)

        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        td4 = self.sepconv_td4(self.fuse_td4(p4, self._resize_to(p5, p4)))
        td3 = self.sepconv_td3(self.fuse_td3(p3, self._resize_to(td4, p3)))

        bu4 = self.sepconv_bu4(self.fuse_bu4(
            p4,
            td4,
            F.max_pool2d(td3, 2)
        ))

        bu5 = self.sepconv_bu5(self.fuse_bu5(
            p5,
            F.max_pool2d(td4, 2) if td4.shape[-2:] != p5.shape[-2:] else td4,
            F.max_pool2d(bu4, 2) if bu4.shape[-2:] != p5.shape[-2:] else bu4
        ))

        return [td3, bu4, bu5, p6, p7]





class Conv2dStaticSamePadding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
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
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x
    

class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

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


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats
    
    
    

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=128):
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

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
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
