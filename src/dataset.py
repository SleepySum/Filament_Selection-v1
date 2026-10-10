import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils
from typing import Dict, List, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset for Solar Filament Segmentation Challenge 2026.
    Handles multi-annotator COCO JSON, solar limb masking, CLAHE, and polygon-to-mask conversion.
    """
    
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        transform: Optional[A.Compose] = None,
        mode: str = 'train',
        img_size: int = 2048,
        apply_clahe: bool = True,
        apply_limb_mask: bool = True,
        min_filament_area: int = 150,
    ):
        """
        Args:
            json_path: Path to COCO annotations JSON
            images_dir: Directory containing JPEG images
            transform: Albumentations transform pipeline
            mode: 'train' or 'val'
            img_size: Image size (default 2048)
            apply_clahe: Whether to apply CLAHE enhancement
            apply_limb_mask: Whether to apply solar limb masking
            min_filament_area: Minimum filament area in pixels to keep
        """
        self.json_path = json_path
        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode
        self.img_size = img_size
        self.apply_clahe = apply_clahe
        self.apply_limb_mask = apply_limb_mask
        self.min_filament_area = min_filament_area
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image_id to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        
        # Build list of valid image IDs (each annotator entry is a separate sample)
        self.image_infos = []
        for img_info in self.coco_data['images']:
            img_id = img_info['id']
            # Only include images that have annotations
            if img_id in self.image_id_to_annotations:
                self.image_infos.append(img_info)
        
        # Setup CLAHE
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default albumentations transforms."""
        if self.mode == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),
                A.GridDistortion(p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
    
    def _detect_solar_limb(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Detect solar limb using Hough Circle Transform.
        
        Args:
            image: Grayscale image (H, W)
            
        Returns:
            (center_x, center_y, radius)
        """
        # Apply Gaussian blur for circle detection
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=800,
            maxRadius=1100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Return the largest circle (most likely solar limb)
            largest_circle = max(circles, key=lambda c: c[2])
            return largest_circle[0], largest_circle[1], largest_circle[2]
        else:
            # Fallback: assume center of image
            h, w = image.shape
            return w // 2, h // 2, min(w, h) // 2 - 50
    
    def _apply_limb_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply solar limb mask to zero out deep-space pixels.
        
        Args:
            image: Grayscale image (H, W)
            
        Returns:
            Masked image
        """
        h, w = image.shape
        cx, cy, r = self._detect_solar_limb(image)
        
        # Create circular mask
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        
        # Apply mask
        masked_image = image.copy()
        masked_image[~mask] = 0
        
        return masked_image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to enhance filament barbs contrast.
        
        Args:
            image: Grayscale image (H, W)
            
        Returns:
            CLAHE enhanced image
        """
        return self.clahe.apply(image)
    
    def _polygons_to_mask(self, annotations: List[Dict], height: int, width: int) -> np.ndarray:
        """
        Convert COCO polygon annotations to instance masks.
        
        Args:
            annotations: List of COCO annotation dicts
            height: Image height
            width: Image width
            
        Returns:
            Instance mask array (N, H, W) where N is number of instances
        """
        masks = []
        for ann in annotations:
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):
                # Polygon format
                rle = mask_utils.frPyObjects(segmentation, height, width)
                if isinstance(rle, list):
                    rle = mask_utils.merge(rle)
                binary_mask = mask_utils.decode(rle)
            else:
                # RLE format
                binary_mask = mask_utils.decode(segmentation)
            
            if binary_mask.sum() >= self.min_filament_area:
                masks.append(binary_mask)
        
        if masks:
            return np.stack(masks, axis=0).astype(np.uint8)
        else:
            return np.zeros((0, height, width), dtype=np.uint8)
    
    def _load_image(self, file_name: str) -> np.ndarray:
        """Load image from disk."""
        img_path = os.path.join(self.images_dir, file_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return image
    
    def __len__(self) -> int:
        return len(self.image_infos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_info = self.image_infos[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Load image
        image = self._load_image(file_name)
        h, w = image.shape
        
        # Apply solar limb masking
        if self.apply_limb_mask:
            image = self._apply_limb_mask(image)
        
        # Apply CLAHE
        if self.apply_clahe:
            image = self._apply_clahe(image)
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(img_id, [])
        
        # Convert polygons to instance masks
        instance_masks = self._polygons_to_mask(annotations, h, w)
        num_instances = instance_masks.shape[0]
        
        # Combine instance masks into single mask for segmentation (class 1 for all filaments)
        if num_instances > 0:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(num_instances):
                combined_mask[instance_masks[i] > 0] = i + 1  # Instance IDs start from 1
        else:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert grayscale to 3-channel for backbone compatibility
        image_3ch = np.stack([image, image, image], axis=-1)
        
        # Apply augmentations
        if self.transform:
            # Prepare mask for albumentations (use combined mask)
            transformed = self.transform(image=image_3ch, mask=combined_mask)
            image_3ch = transformed['image']
            combined_mask = transformed['mask']
        
        # Reconstruct instance masks from combined mask after transform
        # This is needed because transforms may change mask values
        instance_masks_out = []
        if combined_mask.max() > 0:
            for i in range(1, int(combined_mask.max()) + 1):
                inst_mask = (combined_mask == i).astype(np.uint8)
                if inst_mask.sum() >= self.min_filament_area:
                    instance_masks_out.append(inst_mask)
        
        if instance_masks_out:
            instance_masks_out = np.stack(instance_masks_out, axis=0)
        else:
            instance_masks_out = np.zeros((0, h, w), dtype=np.uint8)
        
        # Prepare target dict
        target = {
            'masks': torch.from_numpy(instance_masks_out).float(),
            'labels': torch.ones(instance_masks_out.shape[0], dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'orig_image_id': img_id,
            'file_name': file_name,
        }
        
        return {
            'image': image_3ch,
            'target': target,
        }


def collate_fn(batch):
    """Custom collate function for variable number of instances."""
    images = torch.stack([item['image'] for item in batch])
    targets = [item['target'] for item in batch]
    return images, targets


def get_dataloader(
    json_path: str,
    images_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    mode: str = 'train',
    **dataset_kwargs
) -> DataLoader:
    """Create DataLoader for SolarFilamentDataset."""
    dataset = SolarFilamentDataset(
        json_path=json_path,
        images_dir=images_dir,
        mode=mode,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# Utility function to split dataset by image_id for validation
def create_train_val_split(
    json_path: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset by unique image files (not annotator entries).
    
    Args:
        json_path: Path to COCO JSON
        val_ratio: Validation split ratio
        seed: Random seed
        
    Returns:
        (train_image_infos, val_image_infos)
    """
    import random
    random.seed(seed)
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Group by base filename (without annotator prefix)
    file_to_entries = {}
    for img_info in coco_data['images']:
        # Extract base filename from image_id
        # Format: annotator_id-filename_without_ext
        parts = img_info['id'].split('-', 1)
        if len(parts) == 2:
            base_name = parts[1]
        else:
            base_name = img_info['id']
        
        if base_name not in file_to_entries:
            file_to_entries[base_name] = []
        file_to_entries[base_name].append(img_info)
    
    # Split base filenames
    base_names = list(file_to_entries.keys())
    random.shuffle(base_names)
    split_idx = int(len(base_names) * (1 - val_ratio))
    train_bases = set(base_names[:split_idx])
    val_bases = set(base_names[split_idx:])
    
    train_infos = []
    val_infos = []
    
    for base_name, entries in file_to_entries.items():
        if base_name in train_bases:
            train_infos.extend(entries)
        else:
            val_infos.extend(entries)
    
    return train_infos, val_infos