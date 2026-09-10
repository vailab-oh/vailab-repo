# TriV-Net-UAV-Perception

Official implementation and reproducibility materials for TriV-Net.

TriV-Net+ is a multi-task visual-perception network for UAV imagery. Given one
RGB image, it predicts semantic segmentation, monocular depth, and object
detections. This repository provides ready-to-run multi-task learning (MTL) and
single-task learning (STL) configurations for the TartanAir Old Town and
Neighborhood environments.

## Repository contents

- [Object-detection preprocessing](OD_preprocessing/): generates bounding-box annotations from preprocessed TartanAir semantic segmentation masks for the Neighborhood and Old Town environments.
- `train.py`: training entry point.
- `train_model/`: TriV-Net+ model, data loader, task losses, GradNorm, and OD encoding/decoding.
- `configs/`: MTL and STL configurations for both environments.
- `scripts/`: eight independent training launchers.
- `dataset_metadata/`: semantic train-ID mappings.
- `evaluation/`: segmentation, depth, and detection metrics.
- `tests/`: unit tests for the core training components.

## Model

| Component | Implementation |
| --- | --- |
| Input | One RGB image resized to 384 x 640 |
| Backbone | EfficientNetV2-S |
| Shared feature network | FPN with 128 channels |
| Semantic segmentation | DeepLabV3+ head |
| Depth estimation | Single-scale FPN depth head |
| Object detection | YOLOv8-style anchor-free head |
| MTL loss weighting | GradNorm |
| OD box representation | Signed LTRB offsets from the grid-cell center |

The MTL configuration enables all three task heads. Each STL configuration uses
the same backbone with only one task head enabled. The corrected OD
implementation is identified as `grid_center_signed_ltrb_v2` in the provided
configuration files.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/vailab-oh/vailab-repo.git
cd vailab-repo/UxV/Vision-AI/TriV-Net-UAV-Perception
```

Python 3.12 is recommended. Create an isolated environment and install the
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the CUDA version on
your system.

## Dataset preparation

The datasets are not included in this repository. Download and preprocess the
TartanAir Old Town and Neighborhood data, then place the files in the following
layout:

```text
data/
  oldtown/
    train/<scene>/image_left/*_left.png
    train/<scene>/seg_train_id/*_left_train_id.png
    train/<scene>/depth_left/*_left_depth.npy
    train/<scene>/annotiation/*_left_det.json
    val/<scene>/...
  neighborhood/
    train/<scene>/image_left/*_left.png
    train/<scene>/seg_train_id/*_left_train_id.png
    train/<scene>/depth_left/*_left_depth.npy
    train/<scene>/annotiation/*_left_det.json
    val/<scene>/...
```

The directory name `annotiation` is intentionally retained for compatibility
with the processed datasets. Each OD JSON record uses a zero-based `class_id`
and a `bbox` in `[x, y, width, height]` format.

The selected training mode determines which files are required:

- MTL: RGB, semantic segmentation, depth, and OD annotations
- SS STL: RGB and semantic train-ID masks
- DE STL: RGB and depth arrays
- OD STL: RGB and OD JSON annotations

Semantic class mappings are provided in `dataset_metadata/`.

## Object-detection preprocessing

The preprocessing code extracts 8-connected semantic components, converts them to axis-aligned bounding boxes, removes extremely small boxes, and writes one JSON annotation file per image.

See the [preprocessing documentation](OD_preprocessing/README.md) for input requirements, class mappings, directory layout, execution commands, and output format.

After preparing the semantic train-ID masks, generate the OD annotations with:

```bash
python OD_preprocessing/generate_oldtown_od_annotations.py
python OD_preprocessing/generate_neighborhood_od_annotations.py
```

## Run training

Each shell script starts one independent training run. The provided
configurations use 100 epochs by default.

Old Town:

```bash
# MTL: semantic segmentation + depth estimation + object detection
bash scripts/train_oldtown_mtl.sh

# STL
bash scripts/train_oldtown_stl_seg.sh
bash scripts/train_oldtown_stl_depth.sh
bash scripts/train_oldtown_stl_od.sh
```

Neighborhood:

```bash
# MTL: semantic segmentation + depth estimation + object detection
bash scripts/train_neighborhood_mtl.sh

# STL
bash scripts/train_neighborhood_stl_seg.sh
bash scripts/train_neighborhood_stl_depth.sh
bash scripts/train_neighborhood_stl_od.sh
```

The scripts use GPU 0 by default. Select another GPU with `GPU_ID`:

```bash
GPU_ID=1 bash scripts/train_neighborhood_mtl.sh
```

To use a specific Python executable, set `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python bash scripts/train_oldtown_mtl.sh
```

Dataset paths, epoch counts, batch sizes, and other hyperparameters can be
changed in `configs/oldtown/` and `configs/neighborhood/`.

## Training outputs

Each run creates a timestamped directory under `Training_result/` containing:

- `best.pth`: best checkpoint
- `last.pth`: final-epoch checkpoint
- `config.yaml`: configuration used for the run
- `metrics.yaml`: validation metrics for each epoch
- `losses.yaml`: raw and weighted task losses
- `loss_balance.yaml`: task weights and loss-balancing settings
- `model_info.yaml`: model and checkpoint information

The reported metrics include:

- Semantic segmentation: mIoU, Dice, and pixel accuracy
- Depth estimation: AbsRel, RMSE, RMSE-log, and delta accuracy
- Object detection: AP50 and AP50-95

## Testing

Run the unit tests from the project root:

```bash
pytest -q
```

## Contact

- **Taegeun Oh, Ph.D. (Principal Investigator):** [tgoh@du.ac.kr](mailto:tgoh@du.ac.kr)
- **Sungjun Jang (Student Researcher):** [charlie1214@khu.ac.kr](mailto:charlie1214@khu.ac.kr)
