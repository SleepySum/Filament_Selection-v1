import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def mask_nms(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4,
    min_size: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-Maximum Suppression for instance masks.
    
    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        min_size: Minimum mask area in pixels
        
    Returns:
        Filtered masks and scores
    """
    if len(masks) == 0:
        return masks, scores
    
    # Filter by minimum size
    areas = masks.sum(axis=(1, 2))
    keep_size = areas >= min_size
    masks = masks[keep_size]
    scores = scores[keep_size]
    
    if len(masks) == 0:
        return masks, scores
    
    # Sort by score descending
    order = scores.argsort()[::-1]
    masks = masks[order]
    scores = scores[order]
    
    # Compute pairwise IoU
    N = len(masks)
    masks_flat = masks.reshape(N, -1).astype(np.float32)
    
    # Intersection
    inter = np.dot(masks_flat, masks_flat.T)
    
    # Areas
    areas = masks_flat.sum(axis=1)
    
    # Union
    union = areas[:, None] + areas[None, :] - inter
    
    # IoU
    iou = inter / (union + 1e-6)
    
    # NMS
    keep = []
    suppressed = np.zeros(N, dtype=bool)
    
    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(i)
        # Suppress overlapping masks
        suppressed[iou[i] > iou_threshold] = True
        suppressed[i] = False  # Don't suppress self
    
    return masks[keep], scores[keep]


def watershed_split(
    mask: np.ndarray,
    min_distance: int = 10,
    threshold_rel: float = 0.3,
) -> List[np.ndarray]:
    """
    Split over-merged filament using watershed on distance transform.
    
    Args:
        mask: Binary mask (H, W)
        min_distance: Minimum distance between peaks
        threshold_rel: Relative threshold for peak detection
        
    Returns:
        List of split binary masks
    """
    if mask.sum() < 150:
        return [mask]
    
    # Distance transform
    distance = ndimage.distance_transform_edt(mask)
    
    # Find peaks
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False,
    )
    
    if len(coords) <= 1:
        return [mask]
    
    # Create markers
    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    
    # Watershed
    labels = watershed(-distance, markers, mask=mask)
    
    # Extract individual masks
    split_masks = []
    for i in range(1, labels.max() + 1):
        inst_mask = (labels == i).astype(np.uint8)
        if inst_mask.sum() >= 150:
            split_masks.append(inst_mask)
    
    return split_masks if split_masks else [mask]


def barb_reconnection(
    mask: np.ndarray,
    max_gap: int = 5,
    min_barb_length: int = 10,
) -> np.ndarray:
    """
    Reconnect fragmented filament barbs using morphological operations.
    
    Args:
        mask: Binary mask (H, W)
        max_gap: Maximum gap to bridge
        min_barb_length: Minimum barb length to preserve
        
    Returns:
        Reconnected mask
    """
    # Skeletonize to find filament structure
    skeleton = cv2.ximgproc.thinning((mask * 255).astype(np.uint8))
    skeleton = skeleton > 0
    
    # Find connected components in skeleton
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    
    if num_labels <= 2:  # Background + 1 component
        return mask
    
    # Analyze each component
    main_component = None
    main_size = 0
    barbs = []
    
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        comp_size = comp_mask.sum()
        
        if comp_size > main_size:
            if main_component is not None:
                barbs.append(main_component)
            main_component = comp_mask
            main_size = comp_size
        else:
            barbs.append(comp_mask)
    
    # Try to reconnect barbs to main component
    if main_component is not None and len(barbs) > 0:
        # Dilate main component
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
        main_dilated = cv2.dilate(main_component.astype(np.uint8), kernel)
        
        for barb in barbs:
            # Check if barb touches dilated main component
            if (barb & main_dilated).any():
                # Reconnect by adding barb to main
                main_component = main_component | barb
    
    # Reconstruct mask from reconnected skeleton
    reconnected = main_component.astype(np.uint8) * 255
    reconnected = cv2.dilate(reconnected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    reconnected = reconnected > 0
    
    # Combine with original mask to preserve thickness
    result = mask | reconnected
    
    return result.astype(np.uint8)


def remove_small_instances(
    masks: np.ndarray,
    scores: np.ndarray,
    min_size: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove instances below minimum size.
    
    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        min_size: Minimum area in pixels
        
    Returns:
        Filtered masks and scores
    """
    areas = masks.sum(axis=(1, 2))
    keep = areas >= min_size
    return masks[keep], scores[keep]


def merge_overlapping_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge highly overlapping masks (likely same filament).
    
    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for merging
        
    Returns:
        Merged masks and scores
    """
    if len(masks) <= 1:
        return masks, scores
    
    N = len(masks)
    masks_flat = masks.reshape(N, -1).astype(np.float32)
    
    # Compute IoU
    inter = np.dot(masks_flat, masks_flat.T)
    areas = masks_flat.sum(axis=1)
    union = areas[:, None] + areas[None, :] - inter
    iou = inter / (union + 1e-6)
    
    # Find clusters of overlapping masks
    visited = np.zeros(N, dtype=bool)
    merged_masks = []
    merged_scores = []
    
    for i in range(N):
        if visited[i]:
            continue
        
        # Find all masks overlapping with this one
        cluster = [i]
        visited[i] = True
        
        for j in range(i + 1, N):
            if not visited[j] and iou[i, j] > iou_threshold:
                cluster.append(j)
                visited[j] = True
        
        # Merge cluster
        if len(cluster) == 1:
            merged_masks.append(masks[i])
            merged_scores.append(scores[i])
        else:
            # Weighted merge by scores
            cluster_masks = masks[cluster]
            cluster_scores = scores[cluster]
            
            # Weighted average
            weights = cluster_scores / (cluster_scores.sum() + 1e-6)
            merged_mask = np.zeros_like(masks[0], dtype=np.float32)
            for m, w in zip(cluster_masks, weights):
                merged_mask += m * w
            
            merged_mask = (merged_mask > 0.5).astype(np.uint8)
            merged_score = cluster_scores.max()
            
            merged_masks.append(merged_mask)
            merged_scores.append(merged_score)
    
    return np.array(merged_masks), np.array(merged_scores)


def postprocess_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    image: Optional[np.ndarray] = None,
    config: Dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full postprocessing pipeline for predicted masks.
    
    Args:
        masks: (N, H, W) predicted binary masks
        scores: (N,) confidence scores
        image: Original image for barb reconnection (optional)
        config: Postprocessing configuration
        
    Returns:
        Processed masks and scores
    """
    default_config = {
        'nms_iou_threshold': 0.4,
        'min_size': 150,
        'watershed_min_distance': 10,
        'watershed_threshold_rel': 0.3,
        'merge_iou_threshold': 0.7,
        'barb_reconnect': True,
        'barb_max_gap': 5,
    }
    
    if config:
        default_config.update(config)
    
    cfg = default_config
    
    # 1. Remove small instances
    masks, scores = remove_small_instances(masks, scores, cfg['min_size'])
    
    if len(masks) == 0:
        return masks, scores
    
    # 2. Mask NMS
    masks, scores = mask_nms(masks, scores, cfg['nms_iou_threshold'], cfg['min_size'])
    
    if len(masks) == 0:
        return masks, scores
    
    # 3. Merge overlapping masks
    masks, scores = merge_overlapping_masks(masks, scores, cfg['merge_iou_threshold'])
    
    # 4. Watershed split for over-merged filaments
    split_masks = []
    split_scores = []
    
    for mask, score in zip(masks, scores):
        split = watershed_split(
            mask,
            min_distance=cfg['watershed_min_distance'],
            threshold_rel=cfg['watershed_threshold_rel'],
        )
        split_masks.extend(split)
        split_scores.extend([score] * len(split))
    
    masks = np.array(split_masks)
    scores = np.array(split_scores)
    
    # 5. Barb reconnection
    if cfg['barb_reconnect'] and image is not None:
        reconnected_masks = []
        for mask in masks:
            reconnected = barb_reconnection(
                mask,
                max_gap=cfg['barb_max_gap'],
            )
            reconnected_masks.append(reconnected)
        masks = np.array(reconnected_masks)
    
    # 6. Final size filter
    masks, scores = remove_small_instances(masks, scores, cfg['min_size'])
    
    return masks, scores


def refine_masks_with_crf(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    gaussian_sxy: int = 3,
    gaussian_compat: float = 3,
    bilateral_sxy: int = 50,
    bilateral_srgb: int = 5,
    bilateral_compat: float = 10,
) -> np.ndarray:
    """
    Refine masks using DenseCRF (requires pydensecrf).
    
    Args:
        image: (H, W, 3) RGB image
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        gaussian_sxy: Gaussian pairwise spatial std
        gaussian_compat: Gaussian pairwise compatibility
        bilateral_sxy: Bilateral pairwise spatial std
        bilateral_srgb: Bilateral pairwise color std
        bilateral_compat: Bilateral pairwise compatibility
        
    Returns:
        Refined masks
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf not installed, skipping CRF refinement")
        return masks
    
    H, W = image.shape[:2]
    refined_masks = []
    
    for i, mask in enumerate(masks):
        # Prepare unary potentials
        prob_fg = mask.astype(np.float32) * scores[i]
        prob_bg = 1 - prob_fg
        
        unary = np.stack([prob_bg, prob_fg], axis=0)
        unary = unary.reshape(2, -1)
        unary = -np.log(unary + 1e-8)
        
        # CRF
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary)
        
        # Gaussian pairwise
        d.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)
        
        # Bilateral pairwise
        d.addPairwiseBilateral(
            sxy=bilateral_sxy,
            srgb=bilateral_srgb,
            rgbim=image,
            compat=bilateral_compat,
        )
        
        # Inference
        Q = d.inference(5)
        refined = np.argmax(Q, axis=0).reshape(H, W)
        refined_masks.append(refined.astype(np.uint8))
    
    return np.array(refined_masks)


def postprocess_batch(
    batch_masks: List[np.ndarray],
    batch_scores: List[np.ndarray],
    batch_images: Optional[List[np.ndarray]] = None,
    config: Dict = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Postprocess a batch of predictions.
    
    Args:
        batch_masks: List of (N, H, W) masks per image
        batch_scores: List of (N,) scores per image
        batch_images: List of (H, W, 3) images (optional)
        config: Postprocessing config
        
    Returns:
        Lists of processed masks and scores
    """
    results_masks = []
    results_scores = []
    
    for i, (masks, scores) in enumerate(zip(batch_masks, batch_scores)):
        image = batch_images[i] if batch_images is not None else None
        masks, scores = postprocess_masks(masks, scores, image, config)
        results_masks.append(masks)
        results_scores.append(scores)
    
    return results_masks, results_scores