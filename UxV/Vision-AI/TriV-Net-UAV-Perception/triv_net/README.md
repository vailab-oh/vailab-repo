# TriV-Net

TriV-Net jointly performs object detection (OD), semantic segmentation (SS), and
depth estimation (DE) using a ResNet-50 backbone and a shared FPN.

This package provides **OldTown** training and single-image inference for one
multi-task model (MTL) and three single-task models (STL). The dataset and trained
checkpoints must be provided separately.

## 1. Install

The commands below assume Linux, Bash, and Conda. Clone the `main` branch and
enter the TriV-Net directory. Keep this directory as the working directory for
all subsequent commands:

```bash
git clone --branch main https://github.com/vailab-oh/vailab-repo.git
cd vailab-repo/UxV/Vision-AI/TriV-Net-UAV-Perception/triv_net
conda create -n trivnet python=3.7 pip -y
conda activate trivnet
python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt \
  numpy==1.21.6 Pillow==9.5.0 PyYAML==6.0 \
  opencv-python==4.5.5.64 tqdm==4.65.2
```

If the repository is already downloaded, skip `git clone` and enter the same
TriV-Net directory. These versions match the tested environment.
The CUDA 11.3 build requires a compatible NVIDIA
GPU and driver. For other hardware, select a matching PyTorch/torchvision pair
from the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v1120);
other package versions have not been validated here.

Check GPU availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

For GPU training this should print `True`. ImageNet backbone weights are
downloaded at the start of training if they are not already cached.

## 2. Prepare the dataset

Provide the **processed** OldTown dataset with this structure:

```text
data/oldtown/
├── train/
│   └── <scene>/
│       ├── image_left/000000_left.png
│       ├── seg_train_id/000000_left_train_id.png
│       ├── depth_left/000000_left_depth.npy
│       └── annotiation/000000_left_det.json
└── val/
    └── <scene>/
        ├── image_left/000000_left.png
        ├── seg_train_id/000000_left_train_id.png
        ├── depth_left/000000_left_depth.npy
        └── annotiation/000000_left_det.json
```

Use your actual scene and frame names. The spelling `annotiation` is required.
This package does not download data, remap raw semantic labels, or generate OD
annotations.

| Input | Required format |
| --- | --- |
| RGB | 480 × 640 PNG images before resizing |
| SS | Single-channel PNGs with train IDs 0–11; ignore ID 255 |
| DE | NumPy arrays in meters; valid values are finite and in `(0, 80]` |
| OD | JSON lists with zero-based `class_id` and `bbox: [x, y, width, height]` in source-image pixels |

Use the [OldTown semantic mapping](dataset_metadata/oldtown_seg_mapping.yaml) for
SS labels. OD IDs must follow this order:

| ID | Class |
| --- | --- |
| 0 | Street Light |
| 1 | Traffic Light |
| 2 | Road Sign |
| 3 | Guard Rail |
| 4 | Waste Bin |

Example OD annotation:

```json
[{"class_id": 0, "bbox": [10, 20, 30, 40]}]
```

Images without objects need an annotation file containing `[]`.
MTL requires all four inputs; STL requires RGB and the labels for its task.
Files must share the same frame stem.

If the dataset is stored elsewhere, edit `dataset_root` in each configuration
you intend to run, for example:

```yaml
dataset_root: /absolute/path/to/oldtown
```

The final directory name must be `oldtown`. Keep training and validation scenes
separate; this package does not create an independent test split.

## 3. Train

Choose one command:

| Model | Command |
| --- | --- |
| MTL: OD + SS + DE | `GPU_ID=0 bash scripts/train_oldtown_mtl.sh` |
| OD only | `GPU_ID=0 bash scripts/train_oldtown_stl_od.sh` |
| SS only | `GPU_ID=0 bash scripts/train_oldtown_stl_seg.sh` |
| DE only | `GPU_ID=0 bash scripts/train_oldtown_stl_depth.sh` |

Change `GPU_ID=0` to an available GPU index. To train all four independent
models sequentially, in SS → DE → OD → MTL order:

```bash
GPU_ID=0 bash scripts/train_original_trivnet_4.sh
```

Each command starts a **new run** from ImageNet initialization. Checkpoint
resume is not implemented.

The YAML files in [configs/original_trivnet/](configs/original_trivnet/) control
training. Defaults are 50 epochs, batch size 4, 384 × 640 inputs, and AdamW with
learning rate 0.0001 and weight decay 1e-7. Change `epochs`, `batch_size`,
`workers`, or `output_root` there as needed. Input height and width must be
multiples of 128.

Training prints the output directory and one summary per epoch. Interactive
terminals also show a progress bar. To suppress informational output:

```bash
GPU_ID=0 bash scripts/train_oldtown_mtl.sh --quiet
```

To check launch commands without starting training:

```bash
DRY_RUN=1 bash scripts/train_original_trivnet_4.sh
```

The direct Python entry point is also available:

```bash
python train.py --config configs/original_trivnet/oldtown_original_trivnet_mtl.yaml
```

### Fixed task weights

MTL uses the fixed, unnormalized weights from Section 3.4.1, Eq. (8):

```text
L_total = 30 × L_OD + 1 × L_SS + 0.3 × L_DE
```

OD uses focal classification plus Smooth L1 regression, SS uses cross-entropy,
and DE uses SiLog with variance focus 0.85. Task weights are not learned.
Each STL model uses its single task loss with weight 1.

## 4. Read the results

Validation runs automatically after each training epoch. The output directory
printed by the trainer contains:

| File | Contents |
| --- | --- |
| `best.pth` | Model parameters selected on validation |
| `last.pth` | Final model parameters |
| `config.yaml` | Configuration used for this run |
| `metrics.yaml` | Per-epoch metrics |
| `losses.yaml` | Per-epoch task losses and weighted losses |
| `model_info.yaml` | Selected epoch and its validation metrics |

Metrics include OD AP50/AP50–95, SS mIoU/Dice/pixel accuracy, and DE
AbsRel/RMSE/RMSE-log/delta1. MTL selects the minimum weighted validation loss;
OD STL selects maximum AP50–95, SS STL maximum mIoU, and DE STL minimum AbsRel.

Use the epoch recorded in `model_info.yaml` when reporting `best.pth` results.
Validation metrics are not independent test results.

## 5. Predict on an image

After training, replace the run and image paths below with your own:

```bash
python predict.py \
  --run-dir Training_result/YOUR_RUN \
  --image data/oldtown/val/YOUR_SCENE/image_left/YOUR_FRAME_left.png \
  --output-dir predictions/example_01 \
  --device cuda:0
```

`--run-dir` must contain both `config.yaml` and `best.pth`. Load only trusted
checkpoints with matching task heads and class counts. Use `--device cpu` if no
GPU is available. Choose a new output directory for each prediction.

The predictor saves only the enabled task outputs:

| Files | Contents |
| --- | --- |
| `input.png` | Resized input |
| `seg.png`, `seg_train_id.png` | Semantic visualization and train-ID mask |
| `depth.png`, `depth_m.npy` | Depth visualization and depth in meters |
| `od.png`, `detections.json` | Detection visualization and decoded boxes |
| `prediction.json` | Input path and prediction settings |

SS and OD colors are fixed by class ID. Depth uses a fixed 0–80 m color range.
Detection display defaults to confidence 0.25, NMS IoU 0.5, and top-k 100;
AP evaluation uses confidence 0.001. Box coordinates refer to the resized input.

## Troubleshooting

- **No usable samples:** check `dataset_root`, folder spelling, and matching
  label filenames. Add `strict_samples: true` to the YAML to report missing
  labels instead of skipping incomplete samples.
- **CUDA out of memory:** lower `batch_size` in the selected YAML.
- **DataLoader worker error:** try `workers: 0` to see the underlying error.
- **Prediction output already exists:** choose a different `--output-dir`.
- **Checkpoint shape mismatch:** keep `config.yaml` and `best.pth` from the same run.

## Tests

```bash
python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 \
  python -m pytest -q -p no:cacheprovider
```

Tests use temporary synthetic data on CPU and do not download pretrained weights.

## Contact

- **Taegeun Oh, Ph.D. (Principal Investigator):** [tgoh@du.ac.kr](mailto:tgoh@du.ac.kr)
- **Sungjun Jang (Student Researcher):** [charlie1214@khu.ac.kr](mailto:charlie1214@khu.ac.kr)
