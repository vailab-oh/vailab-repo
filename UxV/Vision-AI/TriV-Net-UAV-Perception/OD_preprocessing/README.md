# Object Detection Annotation Generation

This directory contains preprocessing scripts used to derive object-detection bounding boxes from semantic segmentation masks for the **Neighborhood** and **Old Town** environments of the [TartanAir dataset](https://theairlab.org/tartanair-dataset/).

The scripts convert selected semantic classes into one JSON annotation file per image. Bounding boxes are obtained from 8-connected components in a single-channel segmentation mask.

This is a shared preprocessing resource for the [TriV-Net project](../README.md).
For the available training implementation, see [TriV-Net+](../triv_net_plus/README.md).

## Scripts

- `generate_neighborhood_od_annotations.py`: generates annotations for Pole, House, and Car objects.
- `generate_oldtown_od_annotations.py`: generates annotations for Street Light, Traffic Light, Road Sign, Guard Rail, and Waste Bin objects.

## Requirements

- Python 3.9 or later
- NumPy
- Pillow
- OpenCV

Install the required packages with:

```bash
python -m pip install numpy Pillow opencv-python-headless
```

`opencv-python` may be used instead of `opencv-python-headless` when OpenCV GUI functionality is needed elsewhere in the same environment.

## Input data

These scripts do **not** operate directly on the raw segmentation files distributed with TartanAir. They expect semantic masks that have already been converted into single-channel train-ID PNG files.

Each input mask must:

- be a two-dimensional, single-channel image;
- contain integer semantic train IDs;
- use the class-to-ID mappings listed below; and
- have a filename ending in `_train_id.png`.

For TriV-Net+, run the scripts from `triv_net_plus/`, where its `data/` directory
is located. The expected directory layout is:

```text
TriV-Net-UAV-Perception/
├── OD_preprocessing/
│   ├── README.md
│   ├── generate_neighborhood_od_annotations.py
│   └── generate_oldtown_od_annotations.py
├── triv_net/
│   └── README.md
└── triv_net_plus/
    └── data/
        ├── neighborhood/
        │   ├── train/<scene>/seg_train_id/*_train_id.png
        │   └── val/<scene>/seg_train_id/*_train_id.png
        └── oldtown/
            ├── train/<scene>/seg_train_id/*_train_id.png
            └── val/<scene>/seg_train_id/*_train_id.png
```

The scripts scan every directory matching `data/<environment>/*/*/seg_train_id`. They do not enforce a particular difficulty level, sequence list, or train/validation split.

> **Important:** To make the preprocessing pipeline fully reproducible, publish or document the preceding procedure that converts the original TartanAir segmentation data into `seg_train_id` PNG files.

## Class mappings

### Neighborhood

| Detection ID | Class | Korean name | Segmentation train ID | Minimum box-area ratio |
|---:|---|---|---:|---:|
| 0 | Pole | 전봇대 | 10 | 0.0001 (0.01%) |
| 1 | House | 집 | 1 | 0.001 (0.1%) |
| 2 | Car | 자동차 | 9 | 0.0001 (0.01%) |

Overlapping Car bounding boxes are merged after connected-component extraction. This additional merge step is not applied to Pole or House instances.

### Old Town

| Detection ID | Class | Segmentation train ID | Minimum box-area ratio |
|---:|---|---:|---:|
| 0 | Street Light | 11 | 0.0001 (0.01%) |
| 1 | Traffic Light | 5 | 0.0001 (0.01%) |
| 2 | Road Sign | 2 | 0.0001 (0.01%) |
| 3 | Guard Rail | 7 | 0.0001 (0.01%) |
| 4 | Waste Bin | 6 | 0.0001 (0.01%) |

The area threshold is applied to the **bounding-box area**, not to the number of foreground pixels in a connected component. A box whose area is equal to or smaller than the threshold is discarded.

## Usage

After cloning the repository, move to the `TriV-Net-UAV-Perception/triv_net_plus` directory
so that the relative `data/...` paths used by both preprocessing and training
resolve to the same directory:

```bash
cd vailab-repo/UxV/Vision-AI/TriV-Net-UAV-Perception/triv_net_plus
python ../OD_preprocessing/generate_neighborhood_od_annotations.py
python ../OD_preprocessing/generate_oldtown_od_annotations.py
```

Each script prints per-sequence and overall counts for processed files, generated boxes, and images with no retained annotations.

If a script reports `total files=0`, verify the current working directory and the input directory layout before treating the run as successful.

## Output

For an input named:

```text
000000_left_train_id.png
```

the corresponding output is:

```text
annotiation/000000_left_det.json
```

The output directory name `annotiation` reflects the current implementation. If it is corrected to `annotation` or `annotations`, update both scripts and any downstream data loader at the same time.

Each JSON file contains a list of annotations. Bounding boxes use absolute pixel coordinates in the following format:

```text
[x, y, width, height]
```

Example Neighborhood annotation:

```json
[
  {
    "class_id": 2,
    "class_name": "Car",
    "class_name_ko": "자동차",
    "bbox": [214, 163, 48, 31]
  }
]
```

Example Old Town annotation:

```json
[
  {
    "class_id": 0,
    "class_name": "Street Light",
    "bbox": [301, 82, 19, 117]
  }
]
```

An image with no retained object produces an empty JSON list:

```json
[]
```

## Processing procedure

For each segmentation mask, the scripts:

1. create a binary mask for each target semantic class;
2. find 8-connected components;
3. calculate the axis-aligned bounding box of each component;
4. discard boxes below the configured image-area threshold;
5. merge overlapping Car boxes in the Neighborhood environment; and
6. save the remaining annotations as JSON.

Coordinates are derived using an exclusive maximum boundary, so the saved width and height include every pixel in the connected component.

## Reproducibility notes

- The Neighborhood House threshold is ten times larger than the threshold used for the other classes. Preserve this value only if it matches the reported experiment.
- The code uses `Pole`, while related text may refer to the same class as `Utility Pole`. Use one name consistently in papers, configuration files, and evaluation code.
- Neighborhood annotations contain `class_name_ko`; Old Town annotations do not. Downstream code should not assume that this optional field is always present.
- Existing JSON files are overwritten without prompting.
- The scripts select all matching sequences. Dataset splits must be controlled through the directory contents or by modifying the scripts.

## TartanAir citation

If this preprocessing code is used in a publication, cite the original TartanAir dataset:

```bibtex
@inproceedings{wang2020tartanair,
  title     = {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author    = {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and
               Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and
               Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2020}
}
```

Refer to the [official TartanAir repository](https://github.com/castacks/tartanair_tools) and dataset website for download instructions and dataset terms.
