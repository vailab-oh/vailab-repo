# VAI Lab Research Repository

This repository organizes research resources from VAI Lab at Dong Seoul University.

**Research homepage:** [VAI Lab](https://vailab-oh.github.io/)

## Research Areas

- **UxV**
  - [Vision AI](UxV/Vision-AI/)
    - [XGate-Net](UxV/Vision-AI/XGate-Net/): joint semantic segmentation and depth estimation
    - [TriV-Net / TriV-Net+ UAV Perception](UxV/Vision-AI/TriV-Net-UAV-Perception/)
      - [TriV-Net: OldTown training and inference](UxV/Vision-AI/TriV-Net-UAV-Perception/triv_net/README.md)
      - [TriV-Net+: implementation and training](UxV/Vision-AI/TriV-Net-UAV-Perception/triv_net_plus/README.md)
      - [Shared OD preprocessing: annotation generation](UxV/Vision-AI/TriV-Net-UAV-Perception/OD_preprocessing/README.md)
  - [Path Planning](UxV/Path-Planning/)
    - [TA-RRT*](UxV/Path-Planning/TA-RRT/)
    - [TWA-RRT*](UxV/Path-Planning/TWA-RRT/): time-window-aware path planning; paper accepted, publication details forthcoming.
- **[Medical AI](Medical-AI/)**
  - [Rabbit Brain Histopathology Dataset](Medical-AI/Rabbit-Brain-Histopathology-Dataset/)

## Structure

```text
vailab-repo/
├── UxV/
│   ├── Vision-AI/
│   │   ├── XGate-Net/
│   │   └── TriV-Net-UAV-Perception/
│   │       ├── OD_preprocessing/
│   │       ├── triv_net/
│   │       └── triv_net_plus/  # Available implementation
│   └── Path-Planning/
│       ├── TA-RRT/
│       └── TWA-RRT/
└── Medical-AI/
    └── Rabbit-Brain-Histopathology-Dataset/
```

Project-specific code, models, datasets, and documentation are organized within each research area.

## Contact

- **Taegeun Oh, Ph.D. (Principal Investigator):** [tgoh@du.ac.kr](mailto:tgoh@du.ac.kr)
