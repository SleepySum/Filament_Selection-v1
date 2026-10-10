import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_augmentations(img_size: int = 2048) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        img_size: Input image size
        
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.15,
            rotate_limit=45,
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        
        # Elastic and grid distortions for filament deformation
        A.OneOf([
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.OpticalDistortion(
                distort_limit=0.05,
                shift_limit=0.05,
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
        ], p=0.5),
        
        # Photometric transforms
        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        # Cutout for robustness
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            mask_fill_value=0,
            p=0.3
        ),
        
        # Normalize and to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def get_val_augmentations(img_size: int = 2048) -> A.Compose:
    """
    Get validation augmentation pipeline (only normalize).
    
    Args:
        img_size: Input image size
        
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def get_test_augmentations(img_size: int = 2048) -> A.Compose:
    """
    Get test augmentation pipeline (only normalize, no random augmentations).
    
    Args:
        img_size: Input image size
        
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_tta_augmentations() -> List[A.Compose]:
    """
    Get Test Time Augmentation (TTA) pipelines.
    
    Returns:
        List of augmentation pipelines for TTA
    """
    return [
        # Original
        A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Rotate 90
        A.Compose([
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]


def apply_tta(model, image: torch.Tensor, tta_transforms: List[A.Compose], device: torch.device) -> torch.Tensor:
    """
    Apply Test Time Augmentation and average predictions.
    
    Args:
        model: Trained model
        image: Input image tensor (1, C, H, W)
        tta_transforms: List of TTA augmentation pipelines
        device: Torch device
        
    Returns:
        Averaged prediction
    """
    model.eval()
    predictions = []
    
    # Convert tensor to numpy for albumentations
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.astype(np.uint8)
    
    for transform in tta_transforms:
        # Apply transform
        transformed = transform(image=img_np)
        aug_img = transformed['image'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(aug_img)
        
        predictions.append(pred)
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred