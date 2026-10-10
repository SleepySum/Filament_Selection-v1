import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pycocotools import mask as mask_utils
import warnings
warnings.filterwarnings('ignore')

from src.dataset import SolarFilamentDataset, get_dataloader
from src.models import FilamentUNet, build_model
from src.postprocess import postprocess_masks, postprocess_batch
from src.rle_utils import encode_rle, to_submission_format
from src.augmentations import get_test_augmentations


def load_model(
    model_path: str,
    model_type: str = 'unet',
    device: torch.device = None,
    **model_kwargs
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model type ('unet', 'mask2former', 'solov2')
        device: Torch device
        **model_kwargs: Model arguments
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'unet':
        model = FilamentUNet(**model_kwargs)
    else:
        model = build_model(model_type, **model_kwargs)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def inference_on_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transform,
    device: torch.device,
    postprocess_config: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image: Grayscale image (H, W)
        transform: Albumentations transform
        device: Torch device
        postprocess_config: Postprocessing configuration
        
    Returns:
        Tuple of (masks, scores)
    """
    # Apply limb mask and CLAHE (same as training)
    from src.dataset import SolarFilamentDataset
    dataset = SolarFilamentDataset(
        json_path='',  # Not used
        images_dir='',
        apply_clahe=True,
        apply_limb_mask=True,
    )
    
    # Preprocess
    image = dataset._apply_limb_mask(image)
    image = dataset._apply_clahe(image)
    
    # Convert to 3-channel
    image_3ch = np.stack([image, image, image], axis=-1)
    
    # Transform
    transformed = transform(image=image_3ch)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        if hasattr(model, 'predict'):
            # Models with built-in predict method
            predictions = model.predict(input_tensor)
            masks = predictions[0]['masks']
            scores = predictions[0]['scores']
        else:
            # UNet-style models
            instance_logits, semantic_logits = model(input_tensor)
            
            instance_probs = torch.sigmoid(instance_logits)
            semantic_probs = torch.sigmoid(semantic_logits)
            
            # Combine instance and semantic
            combined = instance_probs * semantic_probs
            combined = combined.squeeze(0).cpu().numpy()  # (num_instances, H, W)
            
            # Threshold
            masks = (combined > 0.5).astype(np.uint8)
            scores = combined.mean(axis=(1, 2))
            
            # Filter by score
            keep = scores > 0.3
            masks = masks[keep]
            scores = scores[keep]
    
    # Postprocess
    if postprocess_config is None:
        postprocess_config = {
            'nms_iou_threshold': 0.4,
            'min_size': 150,
            'watershed_min_distance': 10,
            'watershed_threshold_rel': 0.3,
            'merge_iou_threshold': 0.7,
            'barb_reconnect': True,
            'barb_max_gap': 5,
        }
    
    masks, scores = postprocess_masks(masks, scores, image, postprocess_config)
    
    return masks, scores


def run_test_inference(
    model: torch.nn.Module,
    test_images_dir: str,
    transform,
    device: torch.device,
    postprocess_config: dict = None,
) -> List[Dict]:
    """
    Run inference on all test images.
    
    Args:
        model: Trained model
        test_images_dir: Directory containing test images
        transform: Albumentations transform
        device: Torch device
        postprocess_config: Postprocessing configuration
        
    Returns:
        List of predictions per image
    """
    # Get test image files
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpeg')])
    
    all_predictions = []
    
    for file_name in tqdm(image_files, desc="Running inference"):
        img_path = os.path.join(test_images_dir, file_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Extract image_id from filename (remove .jpeg extension)
        image_id = file_name.replace('.jpeg', '')
        
        # Run inference
        masks, scores = inference_on_image(
            model, image, transform, device, postprocess_config
        )
        
        all_predictions.append({
            'image_id': image_id,
            'file_name': file_name,
            'masks': masks,
            'scores': scores,
        })
    
    return all_predictions


def create_submission(
    predictions: List[Dict],
    output_path: str = 'submission.csv',
) -> pd.DataFrame:
    """
    Create submission CSV from predictions.
    
    Args:
        predictions: List of prediction dicts from run_test_inference
        output_path: Path to save submission.csv
        
    Returns:
        Submission DataFrame
    """
    rows = []
    
    for pred in predictions:
        image_id = pred['image_id']
        masks = pred['masks']
        scores = pred['scores']
        
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            # Encode mask to RLE
            rle = encode_rle(mask)
            
            # Create filament_id
            filament_id = f"{image_id}_{idx}"
            
            # Get RLE counts string (raw, no quotes)
            segmentation_rle = rle['counts']
            
            rows.append({
                'filament_id': filament_id,
                'segmentation_rle': segmentation_rle,
            })
    
    # Create DataFrame
    submission = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
    
    # Save
    submission.to_csv(output_path, index=False, quoting=3)  # quoting=3 means no quotes
    print(f"Submission saved to {output_path}")
    print(f"Total filaments: {len(submission)}")
    
    return submission


def create_submission_from_rles(
    rle_dict: Dict[str, dict],
    output_path: str = 'submission.csv',
) -> pd.DataFrame:
    """
    Create submission from pre-computed RLEs.
    
    Args:
        rle_dict: Dict mapping filament_id to RLE dict
        output_path: Output path
        
    Returns:
        Submission DataFrame
    """
    rows = []
    
    for filament_id, rle in rle_dict.items():
        segmentation_rle = rle['counts']
        rows.append({
            'filament_id': filament_id,
            'segmentation_rle': segmentation_rle,
        })
    
    submission = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
    submission.to_csv(output_path, index=False, quoting=3)
    
    return submission


def validate_submission_format(submission_path: str) -> bool:
    """
    Validate submission.csv format.
    
    Args:
        submission_path: Path to submission.csv
        
    Returns:
        True if valid
    """
    df = pd.read_csv(submission_path)
    
    # Check columns
    required_cols = ['filament_id', 'segmentation_rle']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns. Found: {df.columns.tolist()}")
        return False
    
    # Check filament_id format
    for fid in df['filament_id']:
        if '_' not in fid:
            print(f"Warning: filament_id '{fid}' does not contain underscore")
    
    # Check RLE format
    for rle_str in df['segmentation_rle']:
        if not isinstance(rle_str, str):
            print(f"Error: RLE is not string: {type(rle_str)}")
            return False
    
    print(f"Submission validation passed: {len(df)} rows")
    return True


def main():
    """Main submission pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar Filament Segmentation - Submission Pipeline')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='unet', help='Model type')
    parser.add_argument('--test_dir', type=str, default='filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images', help='Test images directory')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output submission file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type, device)
    
    # Get transforms
    transform = get_test_augmentations()
    
    # Postprocess config
    postprocess_config = {
        'nms_iou_threshold': 0.4,
        'min_size': 150,
        'watershed_min_distance': 10,
        'watershed_threshold_rel': 0.3,
        'merge_iou_threshold': 0.7,
        'barb_reconnect': True,
        'barb_max_gap': 5,
    }
    
    # Run inference
    print("Running test inference...")
    predictions = run_test_inference(
        model, args.test_dir, transform, device, postprocess_config
    )
    
    # Create submission
    print("Creating submission...")
    submission = create_submission(predictions, args.output)
    
    # Validate
    validate_submission_format(args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()