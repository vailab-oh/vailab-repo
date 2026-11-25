"""License note: this file is a port of third-party code and is NOT covered by this
repository's MIT license. ``ModelUNetTinySum`` comes from
https://github.com/ishitsuka-hikaru/SafeUAV (no license stated upstream);
``ModelUNetDilatedConv`` comes from neural-wrappers v0.31
(https://gitlab.com/mihaicristianpirvu/neural-wrappers, WTFPL).

Faithful ports of SafeUAV's two released architectures: ``ModelUNetTinySum``
("Small", ``HVO_small.pkl``) and ``ModelUNetDilatedConv`` ("Large", ``HVO_large.pkl``).

``ModelUNetTinySum`` — ported (not reimplemented-from-guess) from the official
repository ``https://github.com/ishitsuka-hikaru/SafeUAV`` (default branch
``jetson_real``), file ``unet_tiny_sum.py``. Every layer's channel count, kernel size,
stride, padding and dilation below was cross-checked against the actual tensor shapes
found in the official pretrained checkpoint ``HVO_small.pkl`` (38 tensors, 1,029,283
parameters) and matches exactly.

The only change from the original file is dropping the ``NeuralNetworkPyTorch`` base
class (SafeUAV's own save/load plumbing, from the ``neural-wrappers`` dependency this
benchmark does not use) in favor of plain ``nn.Module``.
The forward pass itself, including its input/output NHWC<->NCHW transposes and the
``F.pad`` skip-connection-size compensation, is unchanged from the original.

``ModelUNetDilatedConv`` — ``HVO_large.pkl`` ("SafeUAV-Net-Large", ~24M params per the
paper's Table 5) is **not** built from a class inside the SafeUAV repo itself; its
official ``main.py`` constructs it as
``ModelUNetDilatedConv(dIn=3, dOut=3, numFilters=64, bottleneckMode=
"dilate2_serial_concatenate")`` imported from ``neural_wrappers.models`` (confirmed by
fetching ``main.py``'s own import line and ``getModel()`` body from the upstream repo).
That class's source isn't in any currently-tagged release of ``neural_wrappers`` at
its current layout, but it does exist at tag ``v0.31``
(``neural_wrappers/models/model_unet_dilatedconv.py``, plus its
``model_unet.py``/``upsample.py`` dependencies for ``UNetBlock``/``UpSampleLayer``) —
fetched from gitlab.com/mihaicristianpirvu/neural-wrappers and ported below the same
way as ``ModelUNetTinySum``: flattened to plain ``nn.Conv2d``/``nn.ConvTranspose2d``
(no ``FeedForwardNetwork``/``UpSampleLayer`` wrapper classes), attribute-definition
order preserved exactly (parameter loading in ``_safeuav_checkpoint.py`` is positional,
via ``model.parameters()`` order — this order must match the original nested-module
graph's traversal order, not just produce the right total count). Verified exactly: a
freshly constructed ``ModelUNetDilatedConv(numFilters=64)`` has 44 tensors and
23,895,107 parameters, matching ``HVO_large.pkl`` exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelUNetTinySum(nn.Module):
    """3-stage encoder/decoder U-Net with a 6-layer summed dilated bottleneck.

    Parameterized exactly as the original: ``dIn`` input channels, ``dOut`` output
    channels (3 for the original HVO classification task; adapted to 2 for binary
    SLZ by rebuilding with ``dOut=2`` and transplanting every layer but ``finalConv``
    — see ``_safeuav_checkpoint.py``), ``numFilters`` (``NF``) the base channel width
    (16 for the "Small" variant, matching the official pretrained checkpoint).
    """

    def __init__(self, dIn: int = 3, dOut: int = 3, numFilters: int = 16):
        super().__init__()
        NF = numFilters
        self.down1 = nn.Conv2d(dIn, NF, kernel_size=3, padding=1)
        self.down1pool = nn.Conv2d(NF, NF, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(NF, NF * 2, kernel_size=3, padding=1)
        self.down2pool = nn.Conv2d(NF * 2, NF * 2, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(NF * 2, NF * 4, kernel_size=3, padding=1)
        self.down3pool = nn.Conv2d(NF * 4, NF * 4, kernel_size=3, stride=2, padding=1)

        self.dilate1 = nn.Conv2d(NF * 4, NF * 8, kernel_size=3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(NF * 8, NF * 8, kernel_size=3, padding=2, dilation=2)
        self.dilate3 = nn.Conv2d(NF * 8, NF * 8, kernel_size=3, padding=4, dilation=4)
        self.dilate4 = nn.Conv2d(NF * 8, NF * 8, kernel_size=3, padding=8, dilation=8)
        self.dilate5 = nn.Conv2d(NF * 8, NF * 8, kernel_size=3, padding=16, dilation=16)
        self.dilate6 = nn.Conv2d(NF * 8, NF * 8, kernel_size=3, padding=32, dilation=32)

        self.up3_tr = nn.ConvTranspose2d(NF * 8, NF * 4, kernel_size=3, stride=2, padding=1)
        self.up3 = nn.Conv2d(NF * 4, NF * 4, kernel_size=3, padding=1)
        self.up2_tr = nn.ConvTranspose2d(NF * 4, NF * 2, kernel_size=3, stride=2, padding=1)
        self.up2 = nn.Conv2d(NF * 2, NF * 2, kernel_size=3, padding=1)
        self.up1_tr = nn.ConvTranspose2d(NF * 2, NF, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.Conv2d(NF, NF, kernel_size=3, padding=1)

        self.finalConv = nn.Conv2d(in_channels=NF, out_channels=dOut, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original convention: x is NHWC on entry; converted to NCHW internally and
        # back to NHWC on the way out. Unchanged from the official source.
        x = x.transpose(1, 3).transpose(2, 3)
        y1 = F.relu(self.down1(x))
        y1_pool = F.relu(self.down1pool(y1))
        y2 = F.relu(self.down2(y1_pool))
        y2_pool = F.relu(self.down2pool(y2))
        y3 = F.relu(self.down3(y2_pool))
        y3_pool = F.relu(self.down3pool(y3))

        y_dilate1 = F.relu(self.dilate1(y3_pool))
        y_dilate2 = F.relu(self.dilate2(y_dilate1))
        y_dilate3 = F.relu(self.dilate3(y_dilate2))
        y_dilate4 = F.relu(self.dilate4(y_dilate3))
        y_dilate5 = F.relu(self.dilate5(y_dilate4))
        y_dilate6 = F.relu(self.dilate6(y_dilate5))
        y_dilate_sum = y_dilate1 + y_dilate2 + y_dilate3 + y_dilate4 + y_dilate5 + y_dilate6

        y_up3_tr = F.pad(F.relu(self.up3_tr(y_dilate_sum)), (0, 1, 0, 1)) + y3
        y_up3 = F.relu(self.up3(y_up3_tr))
        y_up2_tr = F.pad(F.relu(self.up2_tr(y_up3)), (0, 1, 0, 1)) + y2
        y_up2 = F.relu(self.up2(y_up2_tr))
        y_up1_tr = F.pad(F.relu(self.up1_tr(y_up2)), (0, 1, 0, 1)) + y1
        y_up1 = F.relu(self.up1(y_up1_tr))

        y_final = self.finalConv(y_up1)
        y_final = y_final.transpose(1, 3).transpose(1, 2)
        return y_final


class ModelUNetDilatedConv(nn.Module):
    """SafeUAV-Net-Large: 3-stage encoder/decoder with a 6-layer *concatenated*
    (not summed, unlike the Small variant) dilated bottleneck, ``numFilters=64`` to
    match the official ``HVO_large.pkl`` checkpoint's ``unet_big_concatenate`` /
    ``dilate2_serial_concatenate`` configuration. See the module docstring for
    provenance; attribute order below matches the original nested-module parameter
    traversal order exactly (required for the positional checkpoint load).
    """

    def __init__(self, dIn: int = 3, dOut: int = 3, numFilters: int = 64):
        super().__init__()
        NF = numFilters
        # -- encoder: 3x (2x[conv3x3 pad1 + relu]) + maxpool(2) ----------------------
        self.down1_conv1 = nn.Conv2d(dIn, NF, kernel_size=3, padding=1)
        self.down1_conv2 = nn.Conv2d(NF, NF, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.down2_conv1 = nn.Conv2d(NF, NF * 2, kernel_size=3, padding=1)
        self.down2_conv2 = nn.Conv2d(NF * 2, NF * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.down3_conv1 = nn.Conv2d(NF * 2, NF * 4, kernel_size=3, padding=1)
        self.down3_conv2 = nn.Conv2d(NF * 4, NF * 4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # -- bottleneck: 6 serially-chained dilated convs (dilation 2**i, i=0..5), ---
        # each dIn->dIn*2 (first one dIn=NF*4, rest dIn=NF*8), concatenated
        # (6*NF*8 channels) then one conv-transpose (k=3,s=2) maps channels back to
        # NF*4 while upsampling spatially 2x, then concatenated with the pre-pool skip
        # (ConcatenateBlock's pattern — see the module docstring).
        self.dilate = nn.ModuleList(
            [
                nn.Conv2d(
                    NF * 4 if i == 0 else NF * 8,
                    NF * 8,
                    kernel_size=3,
                    padding=2**i,
                    dilation=2**i,
                )
                for i in range(6)
            ]
        )
        self.bottleneck_up = nn.ConvTranspose2d(NF * 8 * 6, NF * 4, kernel_size=3, stride=2)

        # -- decoder: 3x (conv-transpose k=3 s=2 + relu + crop(-1,-1) + concat skip), -
        # each followed by a 2-conv UNetBlock -----------------------------------------
        self.upBlock3_conv1 = nn.Conv2d(NF * 8, NF * 4, kernel_size=3, padding=1)
        self.upBlock3_conv2 = nn.Conv2d(NF * 4, NF * 4, kernel_size=3, padding=1)
        self.up2_tr = nn.ConvTranspose2d(NF * 4, NF * 2, kernel_size=3, stride=2)
        self.upBlock2_conv1 = nn.Conv2d(NF * 4, NF * 2, kernel_size=3, padding=1)
        self.upBlock2_conv2 = nn.Conv2d(NF * 2, NF * 2, kernel_size=3, padding=1)
        self.up1_tr = nn.ConvTranspose2d(NF * 2, NF, kernel_size=3, stride=2)
        self.upBlock1_conv1 = nn.Conv2d(NF * 2, NF, kernel_size=3, padding=1)
        self.upBlock1_conv2 = nn.Conv2d(NF, NF, kernel_size=3, padding=1)

        self.finalConv = nn.Conv2d(in_channels=NF, out_channels=dOut, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original convention: NHWC in, NHWC out (same as ModelUNetTinySum).
        x = x.transpose(1, 3).transpose(2, 3)

        y1 = F.relu(self.down1_conv2(F.relu(self.down1_conv1(x))))
        y1_pool = self.pool1(y1)
        y2 = F.relu(self.down2_conv2(F.relu(self.down2_conv1(y1_pool))))
        y2_pool = self.pool2(y2)
        y3 = F.relu(self.down3_conv2(F.relu(self.down3_conv1(y2_pool))))
        y3_pool = self.pool3(y3)

        feats = []
        y = y3_pool
        for conv in self.dilate:
            y = F.relu(conv(y))
            feats.append(y)
        dilate_cat = torch.cat(feats, dim=1)
        bneck_up = F.pad(F.relu(self.bottleneck_up(dilate_cat)), (0, -1, 0, -1))
        y_bottleneck = torch.cat([y3, bneck_up], dim=1)

        y_up3 = F.relu(self.upBlock3_conv2(F.relu(self.upBlock3_conv1(y_bottleneck))))
        up2 = F.pad(F.relu(self.up2_tr(y_up3)), (0, -1, 0, -1))
        y_up2_in = torch.cat([y2, up2], dim=1)
        y_up2 = F.relu(self.upBlock2_conv2(F.relu(self.upBlock2_conv1(y_up2_in))))
        up1 = F.pad(F.relu(self.up1_tr(y_up2)), (0, -1, 0, -1))
        y_up1_in = torch.cat([y1, up1], dim=1)
        y_up1 = F.relu(self.upBlock1_conv2(F.relu(self.upBlock1_conv1(y_up1_in))))

        y_final = self.finalConv(y_up1)
        y_final = y_final.transpose(1, 3).transpose(1, 2)
        return y_final
