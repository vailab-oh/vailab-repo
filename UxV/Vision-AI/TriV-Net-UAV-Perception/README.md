# TriV-Net-UAV-Perception

Official implementation and reproducibility materials for TriV-Net.

## Repository contents

- [Object-detection preprocessing](OD_preprocessing/): generates bounding-box annotations from preprocessed TartanAir semantic segmentation masks for the Neighborhood and Old Town environments.

## Object-detection preprocessing

The preprocessing code extracts 8-connected semantic components, converts them to axis-aligned bounding boxes, removes extremely small boxes, and writes one JSON annotation file per image.

See the [preprocessing documentation](OD_preprocessing/README.md) for input requirements, class mappings, directory layout, execution commands, and output format.
