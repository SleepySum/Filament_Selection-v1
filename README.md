# Solar Filament Segmentation Challenge 2026

End-to-end solution for the IEEE BigData Cup / MAGFiLO Solar Filament Segmentation Challenge 2026.

## Overview

Instance-level segmentation of solar filaments from high-resolution (2048x2048) 8-bit grayscale H-Alpha solar imagery.

**Primary Metric**: Panoptic Quality (PQ) @ IoU threshold 0.5

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── dataset.py          # COCO JSON parser, multi-annotator loader, limb masking, CLAHE
│   ├── augmentations.py    # Albumentations pipeline (rotations, elastic, gamma)
│   ├── models.py           # Instance segmentation backbones (Mask2Former, SOLOv2, UNet)
│   ├── losses.py           # Tversky Loss (α=0.3, β=0.7) + Focal Loss
│   ├── postprocess.py      # Mask NMS, Watershed splitting, Barb reconnection
│   ├── evaluator.py        # PQ evaluator with diagnostic distributions
│   └── rle_utils.py        # RLE encoding/decoding utilities
├── submission_pipeline.py  # End-to-end test inference → submission.csv
├── train.py                # Training script with validation
├── requirements.txt        # Dependencies
├── solution_demo.ipynb     # Reproducible demonstration notebook
└── README.md
```

## Key Features

1. **Multi-annotator handling**: Each `<annotator_id>-<file_name>` treated as independent sample
2. **Solar limb masking**: Hough Circle Detection to zero out deep-space pixels
3. **CLAHE enhancement**: clipLimit=2.0, tileGridSize=(8,8) for barb contrast
4. **Barb-preserving loss**: Tversky (α=0.3, β=0.7) + Focal Loss
5. **Advanced postprocessing**: 
   - Mask NMS (IoU=0.4)
   - Watershed cluster splitting
   - Barb reconnection via morphological operations
6. **Exact PQ replication**: PyTorch matrix multiplication for IoU matrix

## Installation

```bash
pip install -r requirements.txt
```

## Data Structure

```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                 # 2048x2048 8-bit grayscale JPEGs
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json  # COCO JSON
└── test/
    └── test_images/                  # Test set H-Alpha JPEGs
```

## Training

```bash
python train.py \
    --data_dir filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
    --model_type unet \
    --encoder resnet34 \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --output_dir checkpoints \
    --mixed_precision
```

## Inference & Submission

```bash
python submission_pipeline.py \
    --model_path checkpoints/best_model.pth \
    --model_type unet \
    --test_dir filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --output submission.csv
```

## Submission Format

CSV with columns: `filament_id`, `segmentation_rle`

- `filament_id`: `<image_id>_<instance_idx>` (e.g., `010401-20160920230134Lh_0`)
- `segmentation_rle`: Raw RLE counts string (no quotes)

## Models Supported

- **FilamentUNet**: U-Net with instance head (baseline)
- **SOLOv2Filament**: SOLOv2 instance segmenter
- **Mask2FormerFilament**: Mask2Former with Swin-L backbone

## Loss Functions

- `tversky_focal`: Combined Tversky (α=0.3, β=0.7) + Focal (default)
- `tversky`: Pure Tversky Loss
- `focal`: Pure Focal Loss
- `dice`: Dice Loss
- `combo`: Weighted combination

## Evaluation

Local validation computes:
- Panoptic Quality (PQ)
- IoU / Dice distributions
- Fragmentation (1:n) vs Over-merging (m:1) counts
- Diagnostic visualizations

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (recommended)
- See `requirements.txt` for full list

## Citation

If you use this code, please cite the MAGFiLO dataset and IEEE BigData Cup 2026.