import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .utils import *
from .encoder import *
from .decoder import *


class DynamicPyramidFusion5(nn.Module):
    def __init__(self, C=128, hidden=64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(5 * C, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 5, bias=False)
        )

    def forward(self, xs):
        assert len(xs) == 5, "Expected 5-level features"
        pooled = [self.gap(x).flatten(1) for x in xs]
        z = torch.cat(pooled, dim=1)
        logits = self.mlp(z)
        w = torch.softmax(logits, dim=1)
        out = 0
        for i, x in enumerate(xs):
            wi = w[:, i].view(-1, 1, 1, 1)
            out = out + wi * x
        return out


class CrossTaskGatedFusion(nn.Module):
    def __init__(self, C=128, r=8):
        super().__init__()
        self.gs = nn.Sequential(
            nn.Conv2d(C, C // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // r, C, 1, bias=False),
            nn.Sigmoid()
        )
        self.gd = nn.Sequential(
            nn.Conv2d(C, C // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // r, C, 1, bias=False),
            nn.Sigmoid()
        )
        self.mix = nn.Conv2d(C * 2, C, 1, bias=False)

    def forward(self, Fseg, Fdep):
        gs = self.gs(Fdep)
        gd = self.gd(Fseg)
        seg_ref   = Fseg * (1 + gs)
        depth_ref = Fdep * (1 + gd)
        shared    = self.mix(torch.cat([seg_ref, depth_ref], dim=1))
        return seg_ref, depth_ref, shared


class ASPPLite(nn.Module):
    def __init__(self, C=128, rates=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(C, C, 3, padding=r, dilation=r, bias=False) for r in rates
        ])
        self.proj = nn.Conv2d(C * len(rates), C, 1, bias=False)

    def forward(self, x):
        ys = [b(x) for b in self.branches]
        return self.proj(torch.cat(ys, dim=1))


class DepthToSegEdgeGate(nn.Module):
    def __init__(self, C=128, alpha=0.3):
        super().__init__()
        self.reduce = nn.Conv2d(C, 1, 1, bias=False)
        kx = torch.tensor([[[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
        ky = torch.tensor([[[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0)
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, depth_feat):
        m = self.reduce(depth_feat)
        gx = F.conv2d(m, self.sobel_x, padding=1)
        gy = F.conv2d(m, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        gate = torch.sigmoid(mag)
        return gate, self.alpha


class SegToDepthRegionGate(nn.Module):
    def __init__(self, C=128, beta=0.2):
        super().__init__()
        self.predict = nn.Conv2d(C, 1, 1, bias=False)
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, seg_feat):
        mask = torch.sigmoid(self.predict(seg_feat))
        return mask, self.beta



class models(nn.Module):
    def __init__(self, cfg):
        super(models, self).__init__()
        self.cfg = cfg
        camC = 128

        if cfg['model']['encoder'] == 'EfficientNet':
            self.camencode = Encoder_EfficientNet(camC)
            inC = [320, 112, 40]
        elif cfg['model']['encoder'] == 'Densnet':
            self.camencode = Encoder_DenseNet121(camC)
            inC = [1024, 1024, 512]
        else:
            self.camencode = Encoder_Resnet(camC)
            inC = [2048, 1024, 512]

        if cfg['model']['decoder'] == 'FPN':
            self.decoder = FPN(inC, feat_c=camC)
        elif cfg['model']['decoder'] == 'Bi-FPN':
            self.decoder = BiFPN(inC, feat_c=camC)
        else:
            raise ValueError("cfg['model']['decoder'] must be 'FPN' or 'Bi-FPN'")

        if cfg['model_conf']['od']:
            num_classes = 5
            aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
            num_scales = len([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            num_anchors = len(aspect_ratios) * num_scales

            self.regressor = Regressor(
                in_channels=camC,
                num_anchors=num_anchors,
                num_layers=3,
                pyramid_levels=5,
                onnx_export=True
            )
            self.classifier = ClassificationModel(
                num_features_in=camC,
                num_classes=num_classes
            )
            self.anchors = Anchors(anchor_scale=1.25, onnx_export=True)
            self.regressBoxes = BBoxTransform()
            self.clipBoxes = ClipBoxes()

        if cfg['model_conf']['Seg']:
            self.num_seg_classes = cfg['model_conf']['num_seg_classes']

            self.seg_align = nn.ModuleList([
                nn.Identity(),
                nn.Conv2d(camC, camC, 1, bias=False),
                nn.Conv2d(camC, camC, 1, bias=False),
                nn.Conv2d(camC, camC, 1, bias=False),
                nn.Conv2d(camC, camC, 1, bias=False),
            ])
            self.seg_fuse5   = DynamicPyramidFusion5(C=camC, hidden=64)
            self.seg_adapter = nn.Conv2d(camC, camC, 1, bias=False)

            self.seg_output = nn.Sequential(
                nn.Conv2d(camC, 64, kernel_size=3, padding=1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, self.num_seg_classes, kernel_size=1, padding=0),
            )

        if cfg['model_conf']['Depth']:
            self.depth_align = nn.ModuleList([
                nn.Identity(),
                nn.Conv2d(camC, camC, 1, bias=False),
                nn.Conv2d(camC, camC, 1, bias=False),
                nn.Conv2d(camC, camC, 1, bias=False),
                nn.Conv2d(camC, camC, 1, bias=False),
            ])
            self.depth_fuse5   = DynamicPyramidFusion5(C=camC, hidden=64)
            self.depth_adapter = nn.Conv2d(camC, camC, 1, bias=False)

            self.depth_output = nn.Sequential(
                nn.Conv2d(camC, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1, padding=0),
            )

        both_on = cfg['model_conf']['Seg'] and cfg['model_conf']['Depth']
        if both_on:
            self.xgate       = CrossTaskGatedFusion(C=camC, r=8)
            self.shared_ctx  = ASPPLite(C=camC, rates=(1, 2, 4))
            self.edge_gate   = DepthToSegEdgeGate(C=camC, alpha=0.3)
            self.region_gate = SegToDepthRegionGate(C=camC, beta=0.2)

        self.mtl_cfg = {
            "detach_depth_to_seg": True,
            "detach_seg_to_depth": False,
            "use_alpha_beta_warmup": True,
            "alpha_base": 0.3,
            "beta_base": 0.2,
            "warmup_steps": 1000,
            "use_seg_logit_temperature": True,
            "tau_seg": 0.9,
        }
        self.register_buffer("global_step_buf", torch.zeros(1, dtype=torch.long))

    def _maybe_bump_step(self):
        if self.training:
            self.global_step_buf += 1

    def _warmup_scale(self, base):
        if not self.mtl_cfg["use_alpha_beta_warmup"]:
            return base
        steps = int(self.mtl_cfg["warmup_steps"])
        t = float(self.global_step_buf.item()) / max(1, steps)
        t = min(1.0, t)
        return base * t


    def forward(self, x):
        self._maybe_bump_step()

        feats = self.camencode(x)

        if self.cfg['model']['decoder']:
            out_x = self.decoder(feats)
        else:
            raise ValueError("Decoder must be specified")

        P3, P4, P5, P6, P7 = out_x
        tgt_hw = P3.shape[2:]

        def up_to(z, ref_spatial):
            return F.interpolate(z, size=ref_spatial, mode='bilinear', align_corners=True)

        seg = depth = None

        if self.cfg['model_conf']['Seg']:
            s3 = self.seg_align[0](P3)
            s4 = self.seg_align[1](up_to(P4, tgt_hw))
            s5 = self.seg_align[2](up_to(P5, tgt_hw))
            s6 = self.seg_align[3](up_to(P6, tgt_hw))
            s7 = self.seg_align[4](up_to(P7, tgt_hw))
            fused_seg = self.seg_fuse5([s3, s4, s5, s6, s7])

        if self.cfg['model_conf']['Depth']:
            d3 = self.depth_align[0](P3)
            d4 = self.depth_align[1](up_to(P4, tgt_hw))
            d5 = self.depth_align[2](up_to(P5, tgt_hw))
            d6 = self.depth_align[3](up_to(P6, tgt_hw))
            d7 = self.depth_align[4](up_to(P7, tgt_hw))
            fused_depth = self.depth_fuse5([d3, d4, d5, d6, d7])

        both_on = self.cfg['model_conf']['Seg'] and self.cfg['model_conf']['Depth']

        if both_on:
            seg_ref, depth_ref, shared_ref = self.xgate(fused_seg, fused_depth)
            shared_ctx = self.shared_ctx(shared_ref)

            seg_in   = self.seg_adapter(shared_ctx)   + seg_ref
            depth_in = self.depth_adapter(shared_ctx) + depth_ref

            depth_for_edge = depth_in.detach() if self.mtl_cfg["detach_depth_to_seg"] else depth_in
            edge_gate, alpha_param = self.edge_gate(depth_for_edge)
            alpha = self._warmup_scale(float(alpha_param.item()))
            seg_in = seg_in * (1 + alpha * edge_gate)

            seg_for_region = seg_in if not self.mtl_cfg["detach_seg_to_depth"] else seg_in.detach()
            region_mask, beta_param = self.region_gate(seg_for_region)
            beta = self._warmup_scale(float(beta_param.item()))
            depth_in = depth_in * (1 + beta * region_mask)

        elif self.cfg['model_conf']['Seg']:
            seg_in = fused_seg

        elif self.cfg['model_conf']['Depth']:
            depth_in = fused_depth

        classification = None
        regression = None

        if self.cfg['model_conf']['Seg']:
            seg_logits = self.seg_output(seg_in)

            if self.mtl_cfg["use_seg_logit_temperature"]:
                tau = float(self.mtl_cfg["tau_seg"])
                seg = seg_logits / tau
            else:
                seg = seg_logits

        if self.cfg['model_conf']['Depth']:
            depth_logits = self.depth_output(depth_in)
            max_depth = self.cfg['model_conf'].get('max_depth', 80.0)
            depth = torch.sigmoid(depth_logits) * max_depth

        if self.cfg['model_conf'].get('od', False):
            regression = self.regressor(out_x)
            classification = torch.cat([self.classifier(f) for f in out_x], dim=1)

        return seg, depth, classification, regression
