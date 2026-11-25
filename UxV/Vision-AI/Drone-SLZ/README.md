# Safe Landing Zone segmentation on UAVScenes

Code for the paper *Constructing Safe Landing Zone Pseudo-Labels from Multimodal UAV Scene Data*
(under review). It covers the whole pipeline:

1. **Pseudo-label generation** from the UAVScenes `interval5` release: mesh-rendered depth /
   normals → local PCA slope and roughness → semantic hazard veto → UAV footprint filter →
   per-pixel Safe Landing Zone label.
2. **Leave-one-sequence-out (LOSO) splits** with a temporal and a 5 m spatial guard.
3. **Training and evaluation** of four segmentation families through one shared
   dataset / loss / trainer / metric: SegFormer (B0/B3/B5), DeepLabV3 (ResNet-50/101),
   UPerNet (Swin-T/B/L) and SafeUAV-Net (Small/Large).
4. **UAV footprint-radius study** (r = 0.20 / 0.35 / 0.50 m).

## SLZ label convention

| value | meaning |
|---:|---|
| 0 | Unsafe — the surface was evaluated and judged unsafe |
| 1 | Safe — a UAV can land here (landable region, not just feasible centres) |
| 255 | Ignore — geometry could not be computed; excluded from the loss and every metric |

Image size: `(H, W) = (640, 768)`; model logits `[B, 2, 640, 768]`.

## Installation

```bash
git clone https://github.com/vailab-oh/vailab-repo.git
cd vailab-repo/UxV/Vision-AI/Drone-SLZ

conda create -n slz python=3.10 -y && conda activate slz
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run everything from this folder (`Drone-SLZ/`). No `pip install -e .` is needed; every
script adds this folder to `sys.path`.

## 1. Download UAVScenes

UAVScenes is CC BY-NC-SA 4.0 (academic use only) and is **not** redistributed here. From
[huggingface.co/datasets/sijieaaa/UAVScenes](https://huggingface.co/datasets/sijieaaa/UAVScenes)
download (interval = 5 release):

| archive | size | needed |
|---|---:|---|
| `interval5_CAM_LIDAR.zip` | 28.7 GB | yes (images, poses, intrinsics) |
| `interval5_CAM_label.zip` | 1.5 GB | yes (semantic labels) |
| `terra_3dmap_pointcloud_mesh.zip` | 6.8 GB | yes (meshes for depth / normals) |
| `interval5_LIDAR_label.zip` | 0.1 GB | no |

Unpack them under `<UAVSCENES>/raw/`. The generator looks for these layouts:

```
<UAVSCENES>/raw/interval5/CAM_LIDAR/interval5_CAM_LIDAR/interval5_<scene>/...   (or raw/interval5_CAM_LIDAR/interval5_<scene>/...)
<UAVSCENES>/raw/interval5/CAM_LABEL/interval5_CAM_label/<scene>/...             (or raw/interval5_CAM_label/<scene>/...)
<UAVSCENES>/raw/interval5/terra_3dmap_pointcloud_mesh/<family>/...              (or raw/terra_3dmap_pointcloud_mesh/<family>/...)
```

## 2. Prepare the data

```bash
python prepare_data.py --root <UAVSCENES>
```

This generates the labels and installs the paper's splits. The output is
`<UAVSCENES>/prepared/uavscenes_slz/` (6,930 frames); below, `<PREPARED>` is this
directory. The two steps can also run separately with `--stage labels` or `--stage splits`.

The label defaults are the paper settings:

| parameter | value |
|---|---|
| sequences | AMtown03, AMvalley03, HKairport03, HKisland03 (+ the two GNSS_Evening sequences, not used for training) |
| PCA radius | 0.50 m |
| slope / roughness threshold | 6° / 0.10 m |
| UAV radius + safety buffer | 0.35 m + 0.15 m |
| semantic hazard class IDs | 4, 5, 11, 15, 16, 17, 20, 24 |

`splits/loso/` holds the exact frame lists of every run (`manifest_sha256` `4467e82b…` in
`split_info.json`) and the per-fold class weights. The splits stage copies them to
`<PREPARED>/manifests/loso/` and checks that every listed frame exists. Fold *N* holds out
one sequence:

| fold | test sequence | train | val | test |
|---:|---|---:|---:|---:|
| 1 | AMtown03 | 1,536 | 325 | 1,120 |
| 2 | AMvalley03 | 1,671 | 349 | 961 |
| 3 | HKairport03 | 2,098 | 402 | 605 |
| 4 | HKisland03 | 1,979 | 403 | 599 |

Optional QA: `python tools/verify_dataset.py --root <PREPARED> --scheme loso --fold 1 --split all --tasks slz`.

<details><summary>Rebuilding the splits instead of installing them</summary>

```bash
python tools/make_loso_splits.py --root <PREPARED>                    # -> manifests/loso_temporal (30-frame guard)
python tools/apply_spatial_guard.py --root <PREPARED> --d-guard 5.0   # -> manifests/loso (+ 5 m guard)
python tools/compute_class_weights.py --root <PREPARED>
```
The sha256 values in the new `manifests/loso/split_info.json` should match
`splits/loso/split_info.json`.
</details>

## 3. SafeUAV weights

The SafeUAV-Net weights carry no license upstream, so they are not shipped. Download the
"Dataset + snapshots" archive linked from the
[SafeUAV project page](https://sites.google.com/site/aerialimageunderstanding/safeuav-learning-to-estimate-depth-and-safe-landing-areas-for-uavs)
and place the two HVO snapshots at:

```
external/safeuav/checkpoints/HVO_small.pkl   sha256 99e23586694cda456b3ebe53b6c2f1d2d7652592c1bf2b314983cf2da29d2d33
external/safeuav/checkpoints/HVO_large.pkl   sha256 fa1867fcdd558522cd9c8bf1dae7e59cbef02cb5509dbfc0eaf5677c700bc8e6
```

All other pretrained weights download automatically (Hugging Face / torchvision).

## 4. Train and evaluate

Each model has a key (`--model`) and a config (`configs/<key>.yaml`); the ten keys are listed
in [MODELS.md](MODELS.md). One run:

```bash
torchrun --standalone --nproc_per_node=1 train.py --config configs/segformer_b0.yaml \
    --root <PREPARED> --output outputs/segformer_b0/fold1/seed42 --fold 1 --seed 42

python evaluate.py --root <PREPARED> --model segformer_b0 \
    --checkpoint outputs/segformer_b0/fold1/seed42/best_model --fold 1 --split test \
    --save-json outputs/segformer_b0/fold1/seed42/test_metrics.json
```

Recipe (identical for every model): AdamW, lr 1e-4, weight decay 0.01, poly schedule
(power 0.9) with 5 % warm-up, 50 epochs, global batch 8, AMP, gradient clipping 1.0,
horizontal flip + image-only colour jitter (p = 0.5) and Gaussian blur (p = 0.3),
cross-entropy with ignore 255 and the per-fold inverse-frequency Safe weight. The checkpoint
with the best **val** mIoU is kept; the test set is scored once, afterwards.

All runs of the paper (dry run by default; add `--execute` to launch, `--skip-existing` to
resume):

```bash
python tools/run_experiments.py --root <PREPARED> --mode train --nproc-per-node 1   # 120 runs
python tools/run_experiments.py --root <PREPARED> --mode eval                        # test scoring
python tools/summarize_results.py          # -> <exp dir>/results/summary.md (+ CSVs)
```

`train` = 10 models × 4 folds × 3 seeds (42, 1234, 2025). `--model` trains only the given
keys or families, e.g. `--model segformer_b5` or `--model upernet`; `--fold` and `--seed`
filter too. Everything goes to one experiment directory:

```
outputs/experiments_<YYYYMMDD>/
  main/<model>/fold<N>/seed<S>/            best_model/, config.json, history.json, test_metrics.json
  radius/<tag>/<model>/fold<N>/seed<S>/    footprint-radius study (section 5)
  logs/  results/  results_radius/
```

A new `--mode train` starts `outputs/experiments_<today>`; every other mode, and
`--skip-existing`, uses the latest `outputs/experiments_*`. `--exp-dir` picks one explicitly.

## 5. UAV footprint-radius study

Only the UAV radius changes (PCA radius stays 0.50 m, buffer 0.15 m):

```bash
python tools/make_radius_variant.py --src-root <PREPARED> --uav-radius-m 0.50   # -> <PREPARED>_r050
python tools/make_radius_variant.py --src-root <PREPARED> --uav-radius-m 0.20   # -> <PREPARED>_r020
python prepare_data.py --root <UAVSCENES> --stage splits --variant r050
python prepare_data.py --root <UAVSCENES> --stage splits --variant r020

python tools/run_experiments.py --root <PREPARED> --mode radius-train --nproc-per-node 1   # 32 runs
python tools/run_experiments.py --root <PREPARED> --mode radius-eval
python tools/summarize_radius.py --base-root <PREPARED>   # -> <exp dir>/results_radius/
```

## Repository layout

```
prepare_data.py             labels + splits (section 2)
train.py / evaluate.py      one training run / one test scoring
configs/                    one YAML per model key, all with the paper recipe
splits/loso/                frame lists + class weights of the paper
slz_benchmark/
  adapters/                 one adapter per model family (build / forward / save / load) + model keys
  datasets/uavscenes/       dataset, transforms, split registry, LOSO builder
  generation/               label generator: render/, geometry/, slz/, export/, run.py
  engine/                   model-agnostic trainer and evaluator
  metrics/  losses.py
  experiments.py            models, folds, seeds, recipe, radius study, output layout
tools/                      run_experiments.py, summaries, split rebuild, radius variants, QA
```

New models are added as an adapter (`slz_benchmark/adapters/`) plus a model key, never by
branching the trainer or dataset. Every adapter returns logits `[B, 2, H, W]` (channel
0 = Unsafe, 1 = Safe) already resized to the label size.

## License

- **Code:** MIT (`LICENSE`), except `slz_benchmark/adapters/_safeuav_net.py`, which ports
  third-party code and is not covered by the MIT license:
  `ModelUNetTinySum` (SafeUAV-Net-Small) from
  [ishitsuka-hikaru/SafeUAV](https://github.com/ishitsuka-hikaru/SafeUAV) (no license
  stated upstream), and `ModelUNetDilatedConv` (SafeUAV-Net-Large) from
  [neural-wrappers](https://gitlab.com/mihaicristianpirvu/neural-wrappers) v0.31 (WTFPL).
  SafeUAV: Marcu et al., *SafeUAV: Learning to estimate depth and safe landing areas for
  UAVs from synthetic data*, ECCV Workshops 2018.
- **Data:** UAVScenes (Wang et al., ICCV 2025), built on MARS-LVIG, is
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (academic use
  only). This repository does not redistribute any UAVScenes file. The pseudo-labels that
  `prepare_data.py` produces are adapted material of UAVScenes, so CC BY-NC-SA 4.0 applies if
  you share them. `splits/` holds frame lists (scene / frame timestamp / relative paths) and
  per-fold class weights only, with no images or labels.
- **Model weights:** the pretrained weights each model starts from keep their own
  licenses (the SegFormer weights are non-commercial); see [MODELS.md](MODELS.md).
