# Dead Sea Sinkhole Detection — Training & Testing Pipeline

Semantic segmentation of sinkhole precursors in Sentinel-1 SAR interferograms of the Dead Sea coast, using a U-Net (optionally an Attention U-Net) trained on interferogram patches with LiDAR-derived sinkhole polygons as ground truth.

## Pipeline Overview

```
interferograms + LiDAR polygons
        │
        ▼
1. Patch preparation      prepare_intrfrgrm_pathches_v2.py
        │
        ▼
2. Train/val partition    create_intf_partition.py
        │
        ▼
3. Training               train_sinkholes_unet.py
        │
        ├──▶ 4a. Patch-level testing        test.py
        │
        ├──▶ 4b. Full-interferogram testing test_full_intf.py
        │
        └──▶ 5. Inference on new data       predict_new_intf.py
```

## 1. Data Preparation

- **`prepare_intf_coord_dict.py`** — builds `intf_coord.json`, a dictionary of geographic metadata (origin, pixel size, grid dimensions) per interferogram. Needed by all later stages.
- **`prepare_intrfrgrm_pathches_v2.py`** — the main patch generator:
  1. Loads each interferogram and rasterizes the LiDAR sinkhole polygons into a binary ground-truth mask.
  2. Cuts both into fixed-size patches (default 200×100) with a configurable stride.
  3. Saves all patches, plus a separate set of "non-zero" patches (patches whose mask contains at least one sinkhole pixel) used for training.

Helper scripts: `lidar.py` / `untie_mask_polygs.py` (LiDAR ground-truth handling), `check_patches.py` / `clean_patches.py` (patch QA and cleanup).

## 2. Train/Validation Partition

**`create_intf_partition.py`** splits the data into train and validation sets **by interferogram** (so patches from the same interferogram never appear in both sets) and writes the split to a `partition_*.json` file. Existing partitions (`partition_20_05_*.json`) are included in the repo for reproducibility.

## 3. Training

**`train_sinkholes_unet.py`** trains the segmentation network:

```bash
python train_sinkholes_unet.py
```

Main behavior:

- **Model** — standard U-Net (`unet.py`), or Attention U-Net (`attn_unet.py`).
- **Loss** — masked BCE + soft-Dice, where a validity mask excludes no-data pixels from the loss.
- **Temporal context** — optionally stacks the *k* previous interferograms of the same frame as extra input channels (`--add_temporal`), giving the network the deformation history of each pixel.
- **Partition modes** — random by patch, random by interferogram, spatial split, or a preset partition file from step 2.
- **Outputs** — a checkpoint (`*checkpoint_epoch<N>.pth`) per epoch, a log file per job, and optionally a pickled test/validation dataset for later evaluation.

Run `python train_sinkholes_unet.py -h` for the full list of options.

## 4. Testing

### Patch-level (`test.py`)

Loads a saved test dataset (pickle from training) and a checkpoint, and reports Dice score over the test patches:

```bash
python test.py --test_data_path <test_set.pkl> --model <checkpoint.pth>
```

### Full-interferogram (`test_full_intf.py`)

The main evaluation path. For each test interferogram it:

1. Cuts the interferogram into patches and runs the network on all of them.
2. Reconstructs the full-scene prediction map (overlapping patches can be blended with a Hann window).
3. Thresholds the prediction and converts connected regions to polygons (`polygs.py`).
4. Compares predicted polygons against the LiDAR ground truth and reports detection metrics.

`test_full_intf_v2.py` is a newer variant of the same flow. `evaluate_full_intf_output.py` computes precision/recall-style statistics from the saved full-interferogram outputs, and `remove_no_Data_predictions.py` filters out predictions that fall in no-data regions.

## 5. Predicting on New Interferograms

**`predict_new_intf.py`** runs a trained model on interferograms *without* ground truth: it crops the scene to the model grid, reconstructs the full prediction, and exports the predicted sinkhole polygons as a shapefile for GIS use. `predict_new_intf_vA.py` is an alternative version of this script.

## Repository Layout

| File | Role |
|---|---|
| `unet.py`, `unet_parts.py`, `attn_unet.py` | Network architectures |
| `sinkholes_data_loading.py` | PyTorch dataset / data loading |
| `dice_score.py` | Dice metric and loss |
| `evaluate.py` | Validation-time evaluation utilities |
| `get_intf_info.py` | Interferogram metadata & 11-day sequence lookup |
| `polygs.py` | Mask ↔ polygon conversion, pixel ↔ lon/lat |
| `intf_coord.json` | Per-interferogram coordinate dictionary |
| `partition_*.json` | Saved train/val splits |
| `lidar_mask_polygs.*` | LiDAR ground-truth polygons (shapefile) |

## Requirements

Python 3 with: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `rasterio`, `geopandas`, `shapely`, `scikit-image`, `tqdm`.
