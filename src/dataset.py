"""
Dataset for Solar Filament Segmentation Challenge 2026.

Handles:
- Multi-annotator COCO JSON: each ``images[].id`` entry is treated as its own
  independent sample, even if it points at the same underlying JPEG file.
- Multi-annotator mask resolution: each sample gets its *own* per-annotator
  instance masks, which go through the augmentation pipeline together with
  the image so that geometric transforms always stay aligned.
- Solar limb masking via Hough Circle Detection (zeroes deep space).
- CLAHE (clipLimit=2.0, tileGridSize=(8,8)) to boost filament-barb contrast.
- Grayscale -> 3-channel adapter for backbone compatibility.
"""
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from albumentations.core.composition import Compose as AlbumentationsCompose
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

from .augmentations import get_test_augmentations, get_train_augmentations, get_val_augmentations


def _preprocess_image(image: np.ndarray, apply_clahe: bool, apply_limb_mask: bool) -> np.ndarray:
    """Solar limb masking (Hough circle) + CLAHE on a grayscale image."""
    if apply_limb_mask:
        image = _apply_solar_limb_mask(image)
    if apply_clahe:
        image = _apply_clahe(image)
    return image


def _apply_solar_limb_mask(image: np.ndarray) -> np.ndarray:
    """Detect the solar disk via Hough circle transform and black-out deep space."""
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=800,
        maxRadius=1100,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        cx, cy, r = max(circles, key=lambda c: c[2])
    else:
        # Fallback: assume sun is centered, slightly inset
        h, w = image.shape
        cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 50

    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    out = image.copy()
    out[~inside] = 0
    return out


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _polys_to_instance_masks(
    annotations: List[Dict], height: int, width: int, min_area: int = 150
) -> np.ndarray:
    """Convert COCO polygon annotations into a stack of per-instance binary masks.

    All categories (1:Left, 2:Right, 3:Unidentifiable, 4:Ambiguous) map to the
    same single filament class.
    """
    masks = []
    for ann in annotations:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, list):
            # Polygon format
            rles = mask_utils.frPyObjects(seg, height, width)
            if isinstance(rles, list):
                rle = mask_utils.merge(rles)
            else:
                rle = rles
        else:
            # RLE dict format
            rle = seg
        binary = mask_utils.decode(rle)
        if binary.sum() >= min_area:
            masks.append(binary)
    if masks:
        return np.stack(masks, axis=0)
    return np.zeros((0, height, width), dtype=np.uint8)


class SolarFilamentDataset(Dataset):
    """PyTorch Dataset for the MAGFiLO solar filament training split."""

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        transform: Optional[Any] = None,
        mode: str = "train",
        img_size: int = 2048,
        apply_clahe: bool = True,
        apply_limb_mask: bool = True,
        min_filament_area: int = 150,
    ):
        self.json_path = json_path
        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode
        self.img_size = img_size
        self.apply_clahe = apply_clahe
        self.apply_limb_mask = apply_limb_mask
        self.min_filament_area = min_filament_area

        # If no explicit json_path given, defer loading (used for preprocessing only)
        self.coco_data: Optional[Dict] = None
        self.image_id_to_annotations: Dict[Any, List[Dict]] = {}
        self.image_infos: List[Dict] = []

        if json_path:
            with open(json_path, "r") as f:
                self.coco_data = json.load(f)
            # Group annotations by image id
            for ann in self.coco_data["annotations"]:
                self.image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)
            # Every `images[]` entry = one independent sample (multi-annotator)
            for img_info in self.coco_data["images"]:
                if img_info["id"] in self.image_id_to_annotations:
                    self.image_infos.append(img_info)

        if self.transform is None:
            self.transform = self._get_default_transforms()

    def _get_default_transforms(self):
        if self.mode == "train":
            return get_train_augmentations(self.img_size)
        if self.mode == "val":
            return get_val_augmentations(self.img_size)
        return get_test_augmentations(self.img_size)

    def _load_image(self, file_name: str) -> np.ndarray:
        img_path = os.path.join(self.images_dir, file_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return image

    def __len__(self) -> int:
        return len(self.image_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_info = self.image_infos[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        # Load raw grayscale image
        image = self._load_image(file_name)
        h, w = image.shape

        # Solar limb masking + CLAHE (grayscale, single channel)
        image = _preprocess_image(image, self.apply_clahe, self.apply_limb_mask)

        # Stack to 3 channels for backbone compatibility
        image_3ch = np.stack([image, image, image], axis=-1)

        # Per-annotator instance masks — each sample is INDEPENDENT
        annotations = self.image_id_to_annotations.get(img_id, [])
        instance_masks = _polys_to_instance_masks(
            annotations, h, w, self.min_filament_area
        )

        # Pass BOTH mask (semantic, uint8) and masks (list of per-instance uint8)
        # through the same transform so geometry stays aligned:
        semantic_mask = (instance_masks.sum(axis=0) > 0).astype(np.uint8)
        masks_list: List[np.ndarray] = [
            instance_masks[i].astype(np.uint8)
            for i in range(instance_masks.shape[0])
        ]

        if self.transform is not None:
            if self.mode == "train":
                t = self.transform(image=image_3ch, mask=semantic_mask, masks=masks_list)
            else:
                t = self.transform(image=image_3ch, mask=semantic_mask, masks=masks_list)
            image_3ch = t["image"]
            semantic_mask = t["mask"]
            masks_list = [m for m in t["masks"]]

            # Albumentations may emit float arrays after some ops — binarize
            semantic_mask = (np.asarray(semantic_mask) > 0).astype(np.uint8)
            masks_list = [
                (np.asarray(m) > 0).astype(np.uint8) if m.dtype != np.uint8 else m
                for m in masks_list
            ]

        # Drop any instances that fell below the area threshold after aug
        final_masks = [m for m in masks_list if m.sum() >= self.min_filament_area]
        if final_masks:
            instance_masks_out = np.stack(final_masks, axis=0)
        else:
            instance_masks_out = np.zeros((0, *semantic_mask.shape), dtype=np.uint8)

        target = {
            "masks": torch.from_numpy(instance_masks_out).float(),
            "semantic_mask": torch.from_numpy(semantic_mask).float(),
            "labels": torch.ones(instance_masks_out.shape[0], dtype=torch.int64),
            "image_id": img_id,
            "file_name": file_name,
        }

        # Normalize layout to (C, H, W) for torch
        if isinstance(image_3ch, torch.Tensor):
            if image_3ch.shape[0] == 3:  # already CHW from ToTensorV2
                image_3ch = image_3ch.contiguous()
            elif image_3ch.shape[-1] == 3:  # HWC — permute to CHW
                image_3ch = image_3ch.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"Unexpected image tensor shape {image_3ch.shape}")
        else:
            image_3ch = torch.from_numpy(np.ascontiguousarray(image_3ch)).permute(2, 0, 1).contiguous()

        return {"image": image_3ch, "target": target}


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate: stack images, keep variable-size target dicts as a list."""
    images = torch.stack([item["image"] for item in batch])
    targets = [item["target"] for item in batch]
    return images, targets


def get_dataloader(
    json_path: str,
    images_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    mode: str = "train",
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for SolarFilamentDataset."""
    dataset = SolarFilamentDataset(
        json_path=json_path, images_dir=images_dir, mode=mode, **dataset_kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def create_train_val_split(
    json_path: str, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split by *base* image filename (grouping annotator entries), not by id."""
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    annotated_ids = {ann["image_id"] for ann in coco_data["annotations"]}

    # Group entries by underlying file (id format: <annotator_id>-<file_stem>)
    file_to_entries: Dict[str, List[Dict]] = {}
    for img_info in coco_data["images"]:
        if img_info["id"] not in annotated_ids:
            continue
        parts = img_info["id"].split("-", 1)
        base = parts[1] if len(parts) == 2 else img_info["id"]
        file_to_entries.setdefault(base, []).append(img_info)

    bases = sorted(file_to_entries.keys())
    random.seed(seed)
    random.shuffle(bases)
    split = int(len(bases) * (1 - val_ratio))
    train_bases = set(bases[:split])
    val_bases = set(bases[split:])

    train_infos, val_infos = [], []
    for base, entries in file_to_entries.items():
        if base in train_bases:
            train_infos.extend(entries)
        else:
            val_infos.extend(entries)
    return train_infos, val_infos
