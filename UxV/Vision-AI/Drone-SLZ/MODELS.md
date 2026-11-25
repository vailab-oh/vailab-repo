# Models

Every model below is trained with the same recipe (README §4) on the `loso` splits:
4 folds × 3 seeds (42, 1234, 2025). The base size of each family (its first row) is also
trained for the footprint-radius study (r = 0.20 / 0.50 m, seed 42, README §5).

Train any row with `--model <key>` or `--config configs/<key>.yaml`. `--model-name`
overrides the backbone a key builds.

| family | size | `--model` | backbone it builds | initialization | license of the init weights |
|---|---|---|---|---|---|
| SegFormer | B0 | `segformer_b0` | `nvidia/segformer-b0-finetuned-ade-512-512` | ADE20K, Hugging Face | [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) (non-commercial) |
| | B3 | `segformer_b3` | `nvidia/segformer-b3-finetuned-ade-512-512` | ADE20K, Hugging Face | NVIDIA SegFormer license |
| | B5 | `segformer_b5` | `nvidia/segformer-b5-finetuned-ade-640-640` | ADE20K, Hugging Face | NVIDIA SegFormer license |
| DeepLabV3 | ResNet-50 | `deeplab_r50` | `deeplabv3_resnet50` | COCO (VOC labels), torchvision | BSD-3-Clause |
| | ResNet-101 | `deeplab_r101` | `deeplabv3_resnet101` | COCO (VOC labels), torchvision | BSD-3-Clause |
| UPerNet | Swin-T | `upernet_swin_t` | `openmmlab/upernet-swin-tiny` | ADE20K, Hugging Face | MIT |
| | Swin-B | `upernet_swin_b` | `openmmlab/upernet-swin-base` | ADE20K, Hugging Face | MIT |
| | Swin-L | `upernet_swin_l` | `openmmlab/upernet-swin-large` | ADE20K, Hugging Face | MIT |
| SafeUAV-Net | Small | `safeuav_net_small` | `external/safeuav/checkpoints/HVO_small.pkl` | official SafeUAV HVO snapshot | no license stated upstream |
| | Large | `safeuav_net_large` | `external/safeuav/checkpoints/HVO_large.pkl` | official SafeUAV HVO snapshot | no license stated upstream |

Hugging Face and torchvision weights download automatically. The SafeUAV snapshots must be
downloaded by hand (README §3); they are not redistributed here.

Trained weights are not distributed; README §4 and §5 retrain every run of the paper.
