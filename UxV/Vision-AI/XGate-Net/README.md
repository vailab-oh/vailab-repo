# XGate-Net

XGate-Net is a PyTorch model that jointly performs **semantic segmentation** and **depth estimation** from a single RGB image. The default architecture uses DenseNet121 and FPN, with feature fusion across multiple resolutions and gating between tasks.

This repository contains the model definition, training code, and inference code. Segmentation and depth are trained jointly, and you can choose which prediction outputs to save during inference. Training data and trained XGate-Net weights are not included.

## 1. Installation

The setup uses Python 3.12 with the main dependencies listed below. See [requirements.txt](requirements.txt) for the complete list.

| Package | Version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.6.0 |
| torchvision | 0.21.0 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0.86 |

Install the dependencies in a Python 3.12 environment. The commands below assume Linux/Bash and should be run from the project root containing `train.py`.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

For GPU support, use PyTorch 2.6.0 and torchvision 0.21.0 builds compatible with your CUDA environment. Check GPU availability after installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The encoder is initialized with ImageNet pretrained weights. If these weights are not cached locally, they are downloaded when the model is first created, which requires an internet connection. These encoder weights are separate from a trained XGate-Net checkpoint.

## 2. Project Structure

```text
xgate-net/
├── train.py                 # Training, validation, checkpoint saving and resuming
├── train_config.yaml        # Model configuration and input size
├── export_images_mtl.py     # Image inference and output saving
├── requirements.txt         # Dependencies
└── train_model/
    ├── model.py             # XGate-Net model
    ├── encoder.py           # Image feature extraction
    ├── decoder.py           # Feature pyramid construction
    ├── data.py              # Training data loading and preprocessing
    ├── loss.py              # Loss functions
    ├── utils.py             # Model utilities
    └── __init__.py
```

Both training and inference use the model defined in `train_model/model.py`.

## 3. Data Paths

Set `--data-root` to the directory containing the `train/` and `val/` splits. The default is `data/` in the project root.

```text
data/
├── train/
│   ├── rgb/Scene01/sunset/frames/rgb/Camera_0/rgb_00000.jpg
│   ├── label/Scene01/sunset/frames/classSegmentation/Camera_0/classgt_00000.png
│   └── depth/Scene01/sunset/frames/depth/Camera_0/depth_00000.png
└── val/
    ├── rgb/Scene02/sunset/frames/rgb/Camera_0/rgb_00000.jpg
    ├── label/Scene02/sunset/frames/classSegmentation/Camera_0/classgt_00000.png
    └── depth/Scene02/sunset/frames/depth/Camera_0/depth_00000.png
```

Scene and condition names can vary; keep the rest of the directory structure as shown. RGB, segmentation, and depth files are matched by scene, condition, and frame ID. Ensure that the same combination does not appear in both training and validation.

Check the data layout before training:

```bash
python train.py --data-root ./data --check-data
```

For data on external storage, replace `./data` with the actual path. Quote paths containing spaces. `--check-data` verifies file matching and overlap between splits without starting training.

## 4. Input Size and Model Configuration

Configure the model and training input size in [train_config.yaml](train_config.yaml).

```yaml
height: 384
width: 1280
model:
  encoder: Densnet
  decoder: FPN
model_conf:
  Depth: true
  Seg: true
  od: false
  num_seg_classes: 15
```

| Setting | Configuration key or option | Usage |
| --- | --- | --- |
| Input image size | `height`, `width` | Set both values to multiples of 32. |
| Feature extractor | `model.encoder` | The default `Densnet` refers to DenseNet121. Keep this exact spelling. |
| Feature pyramid | `model.decoder` | The default is `FPN`. |
| Joint depth and segmentation training | `model_conf.Depth`, `model_conf.Seg` | The current trainer requires both to be `true`. |
| Inference outputs to save | Inference option `--task` | Choose `both`, `depth`, or `seg`. |

### Adjusting the input size

The default input size is **1280 pixels wide × 384 pixels high**. To start a new training run at a smaller resolution, change both values in the configuration file. For example, use a width of 640 and a height of 192:

```yaml
height: 192
width: 640
```

A smaller input reduces the number of pixels processed but can affect the representation of small objects and boundaries. During inference, set `--width` and `--height` to the input size used for training.

To maintain a separate configuration, copy the default file and edit the copy:

```bash
cp train_config.yaml train_config_custom.yaml
```

Add `--config train_config_custom.yaml` to the training command to use it. Keep `od: false` and `num_seg_classes: 15` with the current trainer.

### Depth output range

The default depth output uses a **numerical scale from 0 to 80**. The related settings are connected as follows:

| Location | Current behavior |
| --- | --- |
| `train_model/data.py` | Converts raw depth values using `raw / 65535 × 80`. |
| `train_model/model.py` | Multiplies the `sigmoid` output by `model_conf.max_depth`, which defaults to `80.0`. |
| `train.py` | Allows only `max_depth=80.0` for training. |
| Inference option `--max_depth` | Sets the upper bound of the model's depth output. The default is `80.0`. |

**The current training depth limit cannot be changed through the configuration file alone.** To train with a limit other than 80, update the data conversion, valid depth mask, model `max_depth`, and trainer range validation together.

Use the same depth range for training and inference. For the default model, specify `--max_depth 80` or omit the option. Changing it to 40 changes the predicted numerical values, so it should not be used to adjust image brightness or contrast. Use `--norm-mode`, `--cmap`, and `--gamma` for visualization, as described below.

Depth predictions follow the conversion above. Without an appropriate unit conversion, the default model's output values should not be interpreted directly as physical distances in meters.

## 5. Training

### Starting a training run

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./data \
  --output-dir Training_result/run01
```

Replace `./data` with the actual data path if needed. Quote paths containing spaces.

The main training options are:

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `train_config.yaml` | Model and input size configuration file. |
| `--data-root` | `data/` | Parent directory of the training and validation splits. |
| `--output-dir` | `Training_result/xgate-net/` | Output directory. |
| `--epochs` | `50` | Total number of training epochs. |
| `--batch-size` | `8` | Batch size. |
| `--workers` | `8` | Number of data loading worker processes. |
| `--lr` | `1e-4` | Initial learning rate. |
| `--weight-decay` | `1e-7` | AdamW weight decay. |
| `--seed` | `42` | Random seed. |
| `--device` | `cuda` if available, otherwise `cpu` | Training device. |

For example, to use a smaller batch size:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./data \
  --output-dir Training_result/run_small_batch \
  --batch-size 2 \
  --workers 4
```

To use multiple GPUs with DataParallel, set a value such as `CUDA_VISIBLE_DEVICES=0,1` and leave `--device` at its default value of `cuda`. Specifying a device such as `--device cuda:0` uses a single GPU.

For a CPU run, use `--device cpu --workers 0 --threads 2`. Run `python train.py --help` to see all available options.

### Training outputs

Training and validation run every epoch, and the following files are saved:

| File | Contents |
| --- | --- |
| `best_loss_model.pth` | Model weights from the epoch with the lowest validation loss. |
| `last_model.pth` | Model weights from the last completed epoch. |
| `last_checkpoint.pth` | Model, optimizer, scheduler, and random states for resuming training. |
| `metrics.csv` | Training and validation losses and metrics for each epoch. |
| `summary.json` | Best and last epoch metrics, parameter count, and runtime. |
| `config.yaml` | Model configuration, training options, and preprocessing information for the run. |
| `train_manifest.csv`, `val_manifest.csv` | Paths and frame IDs of the data used. |
| `source/` | Copies of the training and model source files used for the run. |
| `run.log` | Execution log. |

Segmentation metrics include mIoU, Dice, and Pixel Accuracy. Depth metrics include AbsRel, RMSE, δ1, δ2, δ3, RMSE Log, and Log10. Depth errors follow the numerical scale used during training.

### Resuming an interrupted run

Use the same output directory and original options, and add `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./data \
  --output-dir Training_result/run01 \
  --resume
```

The model configuration, training options, source files, and data manifests must match the saved run. If the original command included options such as `--batch-size 2`, use the same options when resuming. Keep `--epochs` at the original total number of epochs as well.

Training resumes from the epoch after the last completed epoch. Runs that have already completed all epochs are skipped. Use a new `--output-dir` for a new run. If the output directory already exists and `--resume` is not specified, the trainer stops without overwriting it.

## 6. Inference and Saving Results

Inference requires RGB images and trained model weights. Segmentation and depth ground truth are not required.

### Single image

```bash
python export_images_mtl.py \
  --img /path/to/image.jpg \
  --weights Training_result/run01/best_loss_model.pth \
  --encoder Densnet \
  --decoder FPN \
  --width 1280 \
  --height 384 \
  --max_depth 80 \
  --task both \
  --out-dir out_images
```

Match `--encoder`, `--decoder`, `--width`, `--height`, and `--max_depth` to the training configuration. The inference script does not automatically read `train_config.yaml`. Set `--weights` to a checkpoint trained with this model architecture.

### Image directory

Use `--rgb_dir` instead of `--img`:

```bash
python export_images_mtl.py \
  --rgb_dir /path/to/images \
  --weights Training_result/run01/best_loss_model.pth \
  --encoder Densnet \
  --decoder FPN \
  --width 1280 \
  --height 384 \
  --max_depth 80 \
  --task both \
  --out-dir out_images
```

The script processes `.jpg`, `.jpeg`, `.png`, and `.bmp` files directly inside the specified directory. It does not search subdirectories. It uses a GPU if CUDA is available; otherwise, it uses the CPU.

### Saving depth outputs only

Replace `--task both` with `--task depth` to save only depth outputs. Use `--task seg` to save only segmentation outputs.

`--task` selects which outputs to save. The model still predicts both tasks jointly; this option does not switch to a model trained only for depth estimation.

### Adjusting depth visualization

Depth predictions and visualization settings are configured separately. The following options change the appearance of saved images without affecting the predictions saved in `.npy` files.

| Desired output | Inference option | Behavior |
| --- | --- | --- |
| Highlight depth differences within each image | `--norm-mode p2p98` | Uses the 2nd–98th percentiles of each image for the visualization range. This is the default. |
| Use the same visualization range across images | `--norm-mode max` | Uses a fixed range up to `--max_depth`. Keep `--max_depth 80` for the default model. |
| Use the full range within each image | `--norm-mode minmax` | Normalizes using the minimum and maximum depth values in each image. |
| Change the colormap | `--cmap magma` or `--cmap turbo` | Selects the colormap for depth visualization. |
| Save grayscale depth images only | `--only-gray-depth` | Skips color depth images. |
| Brighten midtones | `--gamma 0.7` | Brightens intermediate normalized values. |
| Darken midtones | `--gamma 1.5` | Darkens intermediate normalized values. |
| Save numerical depth outputs | `--save-raw-depth` | Saves a numerical array and a 16-bit PNG in addition to visualization images. |

For example, to compare depth outputs using a fixed range of 0–80 and save numerical predictions:

```bash
python export_images_mtl.py \
  --rgb_dir /path/to/images \
  --weights Training_result/run01/best_loss_model.pth \
  --encoder Densnet \
  --decoder FPN \
  --width 1280 \
  --height 384 \
  --max_depth 80 \
  --task depth \
  --norm-mode max \
  --cmap turbo \
  --gamma 1.0 \
  --save-raw-depth \
  --out-dir out_images
```

With `p2p98` and `minmax`, the visualization range varies between images. To make the same brightness or color represent the same depth value across images, use `max` and keep the colormap and gamma consistent.

Use `--overlay-alpha` to adjust the segmentation overlay opacity. The default is `0.55`. Values closer to 0 show more of the original image; values closer to 1 show more of the segmentation output. Add `--only-seg-mask` to save only the mask for segmentation.

## 7. Inference Output Files

For an input named `image.jpg`, the following files are generated. Some files are omitted depending on `--task` and the output options.

| File | Purpose |
| --- | --- |
| `image_pred_seg_mask.png` | Segmentation output as a color mask. |
| `image_pred_seg_overlay.png` | Segmentation output overlaid on the original image. |
| `image_pred_depth_color.png` | Color depth visualization. |
| `image_pred_depth_gray.png` | 8-bit grayscale depth visualization. |
| `image_pred_depth.npy` | Array of predicted depth values. Requires `--save-raw-depth`. |
| `image_pred_depth_cm16.png` | 16-bit PNG containing predicted depth values × 100. Requires `--save-raw-depth`. |

Use the `.npy` file to read numerical depth predictions:

```python
import numpy as np

depth = np.load("out_images/image_pred_depth.npy")
print("Array shape:", depth.shape)  # (height, width)
print("Minimum / maximum:", depth.min(), depth.max())
```

Color and grayscale PNGs are visualizations. Values in `*_cm16.png` also follow the default model's numerical scale, multiplied by 100. The filename alone does not imply physical distances in centimeters.

To see all available options:

```bash
python train.py --help
python export_images_mtl.py --help
```

## 8. Troubleshooting

| Issue | What to check |
| --- | --- |
| `No RGB images` | Make sure `--data-root` contains `train/` and `val/`, and verify the directory structure down to `Camera_0`. |
| `unmatched frames` | Compare scene, condition, and frame IDs across RGB, segmentation, and depth files to find missing files. |
| `Identical ... IDs in train and val` | Check whether the same frame is included in both splits. |
| CUDA out of memory | Start a new run with a smaller `--batch-size`. To reduce the input size as well, edit `train_config.yaml` and start a new run. |
| Data loader worker error | Run with `--workers 0` to see the underlying data loading error. |
| Output directory already exists | Use `--resume` for an existing run or a different `--output-dir` for a new run. |
| `Resume refused` | Check that the options, including the total epoch count, model configuration, source files, data paths, and manifests match the original run. |

The recorded training loss is `CE + 0.3 × SiLog`, while the validation loss is `CE + SiLog`. Their magnitudes are not directly comparable because the depth loss has different weights.

## 9. Experimental Results

The following tables report ablation and gain sensitivity results on VKITTI2–Sunset. Values are means over 5-fold cross-validation. SS denotes semantic segmentation, and DE denotes depth estimation. Higher values are better for ↑ metrics; lower values are better for ↓ metrics.

### Ablation study

**Ablation study of XGate-Net on VKITTI2–Sunset.**

Each variant removes the indicated component from the full XGate-Net model.

| Model | SS: Dice ↑ | SS: mIoU ↑ | DE: Abs.Rel. ↓ | DE: δ₁ ↑ |
| --- | ---: | ---: | ---: | ---: |
| w/o DPF | 0.4417 | 0.3667 | 0.3680 | 0.6376 |
| w/o ASA | 0.4658 | 0.3919 | 0.3283 | 0.6436 |
| w/o CTGF | 0.4450 | 0.3700 | 0.3235 | 0.6492 |
| w/o ASPP-Lite | 0.4490 | 0.3761 | 0.3362 | 0.6512 |
| w/o edge gate | 0.4549 | 0.3808 | 0.3453 | 0.6418 |
| w/o region gate | 0.4591 | 0.3848 | 0.3583 | 0.6316 |
| XGate-Net (Full) | 0.4635 | 0.3907 | 0.2949 | 0.6942 |

### Gain sensitivity

**Table 5: Gain sensitivity of XGate-Net on VKITTI2–Sunset. Values are 5-fold means**

α₀ controls the edge gate that transfers information from depth features to segmentation features, while β₀ controls the region gate that transfers information from segmentation features to depth features. Their default values are 0.30 and 0.20, respectively, with linear warm-up over the first 1,000 training steps. `half` halves the corresponding default gain, and `zero` sets it to 0.

**(a) Semantic segmentation**

| Configuration | α₀ | β₀ | Dice ↑ | mIoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| XGate-Net | 0.30 | 0.20 | 0.4635 | 0.3907 |
| Edge half | 0.15 | 0.20 | 0.4835 | 0.4089 |
| Edge zero | 0 | 0.20 | 0.4785 | 0.4046 |
| Region half | 0.30 | 0.10 | 0.4845 | 0.4106 |
| Region zero | 0.30 | 0 | 0.4733 | 0.4005 |
| Both zero | 0 | 0 | 0.4791 | 0.4045 |

**(b) Depth estimation**

| Configuration | α₀ | β₀ | Abs.Rel. ↓ | δ₁ ↑ |
| --- | ---: | ---: | ---: | ---: |
| XGate-Net | 0.30 | 0.20 | 0.2949 | 0.6942 |
| Edge half | 0.15 | 0.20 | 0.2860 | 0.6858 |
| Edge zero | 0 | 0.20 | 0.2890 | 0.6900 |
| Region half | 0.30 | 0.10 | 0.2790 | 0.6856 |
| Region zero | 0.30 | 0 | 0.2823 | 0.6923 |
| Both zero | 0 | 0 | 0.2810 | 0.6912 |

We retain the original XGate-Net gain settings of α₀ = 0.30 and β₀ = 0.20. In the gain sensitivity study, this configuration achieved the highest depth accuracy δ₁ among the evaluated settings. Reducing the gains improved segmentation performance and reduced the absolute relative depth error in some configurations, indicating that the benefits of gain adjustment depend on the evaluation metric.
