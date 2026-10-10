"""
Augmentation pipelines for Solar Filament Segmentation Challenge 2026.

Key design decision: because input images are 3x-grayscale and filament masks
come separately, we pass BOTH ``mask`` and ``masks`` (list of per-instance 2D
masks) through the same Albumentations Compose. Albumentations applies a
*single* identical random geometric/photometric op to every item, so the mask
list always stays aligned with the image — fixing the combined-mask-dropout
bug where augmentations could erase thin filament barbs.
"""
from typing import List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _geom_ops() -> List[A.BasicTransform]:
    """Geometric transforms (applied identically to image + all masks)."""
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            shift_limit=0.0625,
            scale_limit=0.15,
            rotate_limit=45,
            p=0.7,
        ),
        A.OneOf(
            [
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            ],
            p=0.5,
        ),
    ]


def _photometric_ops() -> List[A.BasicTransform]:
    """Photometric transforms (affect only the image, masks pass through)."""
    return [
        A.OneOf(
            [
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.3,
        ),
    ]


def _finalize_pipeline(transforms: List[A.BasicTransform]) -> A.Compose:
    """Shared pipeline tail: float conversion + tensor conversion for torch."""
    return A.Compose(
        transforms + [A.ToFloat(max_value=255.0), ToTensorV2()],
    )


def get_train_augmentations(img_size: int = 2048) -> A.Compose:
    """Training pipeline: geometric + photometric ops, dual-mask approach."""
    return A.Compose(
        _geom_ops() + _photometric_ops() + [A.ToFloat(max_value=255.0), ToTensorV2()],
        additional_targets={'mask': 'mask', 'masks': 'masks'},
    )


def get_val_augmentations(img_size: int = 2048) -> A.Compose:
    """Validation pipeline: pass-through with dual-mask support."""
    return A.Compose(
        [A.ToFloat(max_value=255.0), ToTensorV2()],
        additional_targets={'mask': 'mask', 'masks': 'masks'},
    )


def get_test_augmentations(img_size: int = 2048) -> A.Compose:
    """Test pipeline: pass-through, float conversion, no masks required."""
    return A.Compose(
        [A.ToFloat(max_value=255.0), ToTensorV2()],
    )


def get_tta_transforms() -> List[A.Compose]:
    """Test-Time Augmentation transforms (deterministic geometric ops)."""
    return [
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0)]),
    ]


def invert_transform_idx(idx: int, arr):
    """Undo TTA geometric op idx on array (mask or image) for ensembling."""
    if idx == 0:
        return arr[:, ::-1, :]
    if idx == 1:
        return arr[::-1, :, :]
    if idx == 2:
        return arr[::-1, ::-1, :]
    return arr
