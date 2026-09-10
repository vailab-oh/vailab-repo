# TriV-Net UAV Perception

Research code and reproducibility materials for TriV-Net and TriV-Net+.
The two model versions are organized separately; object-detection preprocessing
is maintained as a shared resource.

## Model versions

| Version | Status | Documentation |
| --- | --- | --- |
| TriV-Net | Code coming soon; implementation has not been uploaded yet. | [TriV-Net](triv_net/README.md) |
| TriV-Net+ | Training implementation, MTL/STL configurations, and evaluation utilities are available. | [TriV-Net+](triv_net_plus/README.md) |

The currently available training code belongs to TriV-Net+. See its documentation
for the model architecture, installation, dataset preparation, and training commands.

## Object-detection preprocessing

The shared [OD_preprocessing](OD_preprocessing/README.md) scripts generate
bounding-box annotations from preprocessed TartanAir semantic segmentation masks
for the Neighborhood and Old Town environments. They extract 8-connected
components, convert them to axis-aligned bounding boxes, remove extremely small
boxes, and write one JSON annotation file per image.

The existing preprocessing scripts and detailed documentation are retained in
`OD_preprocessing/`. See the [preprocessing documentation](OD_preprocessing/README.md)
for input requirements, class mappings, directory layout, and output format.

After preparing the semantic train-ID masks under `triv_net_plus/data/`, run:

```bash
cd vailab-repo/UxV/Vision-AI/TriV-Net-UAV-Perception/triv_net_plus
python ../OD_preprocessing/generate_oldtown_od_annotations.py
python ../OD_preprocessing/generate_neighborhood_od_annotations.py
```

## Directory structure

```text
TriV-Net-UAV-Perception/
├── README.md
├── OD_preprocessing/       # Shared object-detection annotation generation
├── triv_net/               # TriV-Net: code coming soon
│   └── README.md
└── triv_net_plus/           # TriV-Net+ implementation
    ├── README.md
    ├── train.py
    ├── train_model/
    ├── configs/
    ├── scripts/
    ├── dataset_metadata/
    ├── evaluation/
    ├── tests/
    └── requirements.txt
```

## Getting started

- [TriV-Net+ installation and training](triv_net_plus/README.md)
- [Shared object-detection preprocessing](OD_preprocessing/README.md)
- [TriV-Net upload status](triv_net/README.md)

For the existing TriV-Net+ configurations, run training and preprocessing from
`triv_net_plus/`. Both use `triv_net_plus/data/` as the dataset directory;
training outputs are written under `triv_net_plus/Training_result/`.

## Contact

- **Taegeun Oh, Ph.D. (Principal Investigator):** [tgoh@du.ac.kr](mailto:tgoh@du.ac.kr)
- **Sungjun Jang (Student Researcher):** [charlie1214@khu.ac.kr](mailto:charlie1214@khu.ac.kr)
