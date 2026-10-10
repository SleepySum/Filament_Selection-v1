# Technical Report Outline — Solar Filament Segmentation Challenge 2026
*(4-page IEEE Overleaf template, per PRD §6 Step 4)*

## 1. Introduction
- Space-weather motivation: filaments ↔ CME prediction, flare warning.
- H-Alpha ground-based imagery: 2048×2048, limb darkening, cloud interference, multi-observatory variability.
- Challenge: instance segmentation under PQ@0.5 with thin, fragmented structures.

## 2. Related Work
- Classical limb detection (Hough circles), CLAHE contrast adaptation.
- Instance seg: Mask R-CNN → Mask2Former / SOLOv2 comparison.
- Topological post-processing: watershed, morphological spine closing.

## 3. Method
### 3.1 Preprocessing
- Solar limb masking (Hough circle → zero out deep space).
- CLAHE (clipLimit=2.0, tile grid 8×8) to surface low-contrast barbs.
- Grayscale → 3-channel adapter for backbone compatibility.
- Multi-annotator handling: each `images[].id` = independent sample instance mask resolution.

### 3.2 Model Architecture
- Primary: UNet(resnet34) baseline w/ Tversky+Focal stem + (Mask2Former/Swin-L as stretch backend).
- Decoder: reverse-skip U-Net with /4,/8,/16,/32 strides to /1 resolution.
- Dual loss: `L = λ₁·L_focal + λ₂·L_tversky` (α=0.3 FP / β=0.7 FN — barb recall emphasis).

### 3.3 Topological Post-Processing
- CCA noise filter (<150 px drop).
- Mask NMS (IoU=0.4).
- Near-identical merge (IoU>0.7).
- Watershed distance-transform split (many-to-one fix).
- Barb reconnection via skeleton+previous-step overlap (1-to-many fix).

## 4. Experiments
### 4.1 Dataset & Protocol
- 707 train JPEGs, 565 unique files, 925+229 train/val annotator samples, 70/30 split (group-annotator-safe).
- Local harness replicates Kaggle PQ@0.5 with torch.matmul IoU matrix.
- Diagnostics: PQ, IoU/Dice histograms, 1:n fragmentation and m:1 over-merge rates.

### 4.2 Ablations (planned)
- Loss ablation: Focal-only vs Tversky-only vs Combined.
- NMS threshold sweep (0.3 / 0.4 / 0.5).
- Watershed on/off and min_distance sweep.
- Barb reconnection on/off (barb recall metric).
- Backbone: resnet34 vs resnet50 vs Swin-L.

### 4.3 Qualitative Results
- Full-image (2048²) predictions, error buckets from diagnostics, postprocess before/after comparisons.

## 5. Conclusion
- Reproducible open-source release (requirements.txt, notebook, seed-pinned pipeline).
- Limitations: extreme multi-annotator disagreement, heavy limb falloff cases.

## 6. Reproducibility Statement
- `verify_pipeline.py` runs 19 deterministic checks end-to-end.
- `train.py` + `submission_pipeline.py` cover raw-data → submission.csv.
- All randomness (split, init, aug) seeded to `seed=42` via `set_seed()`.
