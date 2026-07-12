# Dead Sea Sinkhole Detection — Training & Testing Pipeline

Semantic segmentation of sinkhole precursors in Sentinel-1 SAR interferograms of the Dead Sea coast, using a U-Net (optionally an Attention U-Net) trained on interferogram patches with manually mapped subsidence polygons as ground truth.

## Pipeline Overview

```
interferograms + subsidence polygons
        │
        ▼
1. Patch preparation      prepare_intrfrgrm_pathches.py
        │
        ▼
2. Training               train_sinkholes_unet.py
        │
        ├──▶ 3a. Patch-level testing        test.py
        │
        ├──▶ 3b. Full-interferogram testing test_full_intf.py
        │
        └──▶ 4. Inference on new data       predict_new_intf.py
```

## 1. Data Preparation

- **`prepare_intf_coord_dict.py`** — builds `intf_coord.json`, a dictionary of geographic metadata (origin, pixel size, grid dimensions) per interferogram. Needed by all later stages.
- **`prepare_intrfrgrm_pathches.py`** — the main patch generator:
  1. Loads each interferogram and rasterizes the mapped subsidence polygons (.shp file) into a binary ground-truth mask.
  2. Cuts both into fixed-size patches (default 200×100) with a configurable stride (default 2)
  3. Saves all patches, plus a separate set of "non-zero" patches (patches whose mask contains at least one subsidence pixel) used for training.

## 2. Training

**`train_sinkholes_unet.py`** trains the segmentation network:

```bash
python train_sinkholes_unet.py (+command line args)
```

Main behavior:

- **Model** — standard U-Net (`unet.py`), or Attention U-Net (`attn_unet.py`).
- **Partition modes** — random by patch, random by interferogram, spatial split, or a preset partition file (for intf partition)
- **Temporal context** — optionally stacks the *k* previous interferograms of the same frame as extra input channels (`--add_temporal`), giving the network the deformation history of each pixel.
- **Loss** — masked BCEwl + Dice loss. `--pos_w` gives higher weight to positive (subsidence) pixels
- **Outputs** — a checkpoint (`*checkpoint_epoch<N>.pth`) per epoch, a log file per job, and optionally a pickled test/validation dataset for later evaluation.

Run `python train_sinkholes_unet.py -h` for the full list of options.

## 3. Testing

### Patch-level (`test.py`)

Loads a saved test dataset (pickle from training) and a checkpoint, and reports Dice score, pixel-level precision/recall, and object-level (OL) precision/recall over the test patches. For the OL metrics, predicted objects are matched to ground-truth objects using an overlap threshold (`--th`) and a buffer in pixels (`--b`):

```bash
python test.py --test_data_path <test_set.pkl> --model <checkpoint.pth>
```

### Full-interferogram (`test_full_intf.py`)

The main evaluation path. For each test interferogram it:

1. loads the interferogram patches and runs the model on all of them (only patches inside the LiDAR coverage region — the valid-region mask from `lidar_mask_polygs.shp` — are predicted).
2. Reconstructs the full-scene prediction map (overlapping patches can be blended with a Hann window).
3. Thresholds the prediction and converts connected regions to polygons (`polygs.py`).
4. Saves the reconstructed arrays (interferogram, prediction, GT mask) and the predicted polygons per interferogram. With `--merge_polygs`, all per-intf polygons are also merged into one combined shapefile.

`evaluate_full_intf_output.py` computes precision/recall-style statistics from the saved full-interferogram outputs, and `remove_no_Data_predictions.py` filters out predictions that fall in no-data regions.

## 4. Predicting on New Interferograms

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
| `lidar_mask_polygs.*` | LiDAR coverage polygons — valid-region mask for prediction (shapefile) |

## Requirements

Python 3 with: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `rasterio`, `geopandas`, `shapely`, `scikit-image`, `tqdm`.
