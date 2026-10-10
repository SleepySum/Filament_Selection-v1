"""
Topological post-processing refinement for filament instance masks.

Pipeline (per PRD step 7):
1. Remove small instances (CCA noise filter, Area < 150 px dropped).
2. Mask-level NMS (IoU threshold = 0.4) to eliminate duplicate instances.
3. Merge near-identical masks (over-merge correction).
4. Watershed split of over-merged filaments (many-to-one fix) using
   distance-transform peaks.
5. Barb reconnection (morphological closing + skeleton reconnect).
6. Final size filter.

No dependency on opencv-contrib's ``cv2.ximgproc``; skeletonization uses
``skimage.morphology.skeletonize``.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

warnings_suppressed = False

DEFAULT_CONFIG = {
    "nms_iou_threshold": 0.4,
    "min_size": 150,
    "watershed_min_distance": 10,
    "watershed_threshold_rel": 0.3,
    "merge_iou_threshold": 0.7,
    "barb_reconnect": True,
    "barb_max_gap": 5,
}


def _iou_matrix(masks: np.ndarray) -> np.ndarray:
    """Fast pairwise IoU between a stack of binary masks."""
    N = masks.shape[0]
    flat = masks.reshape(N, -1).astype(np.float32)
    inter = flat @ flat.T
    areas = flat.sum(axis=1)
    union = areas[:, None] + areas[None, :] - inter
    return inter / (union + 1e-6)


def remove_small_instances(
    masks: np.ndarray, scores: np.ndarray, min_size: int = 150
) -> Tuple[np.ndarray, np.ndarray]:
    """CCA noise filter: drop instances whose area < min_size pixels."""
    if masks.size == 0:
        return masks, scores
    keep = masks.sum(axis=(1, 2)) >= min_size
    return masks[keep], scores[keep]


def mask_nms(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4,
    min_size: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask-level Non-Maximum Suppression (IoU threshold = 0.4)."""
    if masks.size == 0:
        return masks, scores
    keep_size = masks.sum(axis=(1, 2)) >= min_size
    masks, scores = masks[keep_size], scores[keep_size]
    if masks.size == 0:
        return masks, scores

    order = np.argsort(-scores)
    masks, scores = masks[order], scores[order]

    iou = _iou_matrix(masks)
    keep = []
    suppressed = np.zeros(len(masks), dtype=bool)
    for i in range(len(masks)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_threshold
        suppressed[i] = False
    return masks[keep], scores[keep]


def merge_overlapping_masks(
    masks: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster high-IoU (near-identical) masks into single consolidated instances."""
    if len(masks) <= 1:
        return masks, scores
    iou = _iou_matrix(masks)
    visited = np.zeros(len(masks), dtype=bool)
    merged_masks, merged_scores = [], []
    for i in range(len(masks)):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, len(masks)):
            if not visited[j] and iou[i, j] > iou_threshold:
                cluster.append(j)
                visited[j] = True
        if len(cluster) == 1:
            merged_masks.append(masks[i])
            merged_scores.append(scores[i])
        else:
            cluster_masks = masks[cluster]
            cluster_scores = scores[cluster]
            weights = cluster_scores / (cluster_scores.sum() + 1e-6)
            weighted = (cluster_masks * weights[:, None, None]).sum(axis=0)
            merged_masks.append((weighted > 0.5).astype(np.uint8))
            merged_scores.append(cluster_scores.max())
    return np.array(merged_masks), np.array(merged_scores)


def watershed_split(
    mask: np.ndarray,
    min_distance: int = 10,
    threshold_rel: float = 0.3,
    min_size: int = 150,
) -> List[np.ndarray]:
    """Split an over-merged filament using a distance-transform watershed."""
    if mask.sum() < 300 or mask.ndim != 2:
        return [mask]
    distance = ndimage.distance_transform_edt(mask.astype(bool))
    coords = peak_local_max(
        distance, min_distance=min_distance, threshold_rel=threshold_rel, exclude_border=False
    )
    if len(coords) <= 1:
        return [mask]
    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    labels = watershed(-distance, markers, mask=mask.astype(np.uint8))
    split = [
        (labels == i).astype(np.uint8)
        for i in range(1, labels.max() + 1)
        if (labels == i).sum() >= min_size
    ]
    return split if split else [mask]


def barb_reconnection(mask: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Reconnect small orphaned components back to the dominant filament body."""
    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bin_mask)
    if num_labels <= 2:
        return bin_mask  # Background + single component, nothing to reconnect

    sizes = ndimage.sum(bin_mask, labels, range(1, num_labels))
    main_label = int(np.argmax(sizes)) + 1
    main = (labels == main_label).astype(np.uint8)
    barbs = [
        (labels == l).astype(np.uint8)
        for l in range(1, num_labels)
        if l != main_label
    ]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
    main_dilated = cv2.dilate(main, kernel)
    out = main.copy()
    for barb in barbs:
        ys, xs = np.where(barb > 0)
        if len(ys) == 0:
            continue
        if main_dilated[ys, xs].any():
            out |= barb

    # Skeletonize the main component to derive a clean spine, then thicken it
    skel = skeletonize(out > 0).astype(np.uint8)
    spine = cv2.dilate(skel * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out |= (spine > 0).astype(np.uint8)
    return out.astype(np.uint8)


def postprocess_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    image: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full postprocess pipeline. ``image`` is optional (size H, W) for barb reconnect."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    masks = np.asarray(masks)
    scores = np.asarray(scores)

    # 0. Handle empty predictions
    if masks.size == 0 or scores.size == 0:
        return np.zeros((0, 2048, 2048), dtype=np.uint8), np.array([])

    # 1. CCA filter (min_size=150 px)
    masks, scores = remove_small_instances(masks, scores, cfg["min_size"])
    if masks.size == 0:
        return masks, scores

    # 2. Mask NMS (IoU=0.4)
    masks, scores = mask_nms(masks, scores, cfg["nms_iou_threshold"], cfg["min_size"])
    if masks.size == 0:
        return masks, scores

    # 3. Merge near-identical masks (over-merge correction)
    masks, scores = merge_overlapping_masks(masks, scores, cfg["merge_iou_threshold"])

    # 4. Watershed split of over-merged filaments
    split_masks, split_scores = [], []
    for m, s in zip(masks, scores):
        for piece in watershed_split(
            m,
            min_distance=cfg["watershed_min_distance"],
            threshold_rel=cfg["watershed_threshold_rel"],
        ):
            split_masks.append(piece)
            split_scores.append(s)
    if split_masks:
        masks = np.array(split_masks)
        scores = np.array(split_scores)
    else:
        masks, scores = np.zeros((0, 2048, 2048), dtype=np.uint8), np.array([])

    # 5. Barb reconnection
    if cfg["barb_reconnect"] and masks.size > 0:
        reconnected = [
            barb_reconnection(m, max_gap=cfg["barb_max_gap"]) for m in masks
        ]
        masks = np.array(reconnected)

    # 6. Final size filter
    masks, scores = remove_small_instances(masks, scores, cfg["min_size"])
    return masks, scores


def postprocess_batch(
    batch_masks: List[np.ndarray],
    batch_scores: List[np.ndarray],
    batch_images: Optional[List[np.ndarray]] = None,
    config: Optional[Dict] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Postprocess a list of per-image (masks, scores) pairs."""
    out_masks, out_scores = [], []
    for i, (m, s) in enumerate(zip(batch_masks, batch_scores)):
        img = batch_images[i] if batch_images is not None else None
        m2, s2 = postprocess_masks(m, s, img, config)
        out_masks.append(m2)
        out_scores.append(s2)
    return out_masks, out_scores


def refine_masks_with_crf(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Optional DenseCRF refinement; silently skipped if pydensecrf is missing."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf not installed, skipping CRF refinement")
        return masks

    if masks.size == 0:
        return masks
    H, W = image.shape[:2]
    refined = []
    for i, mask in enumerate(masks):
        prob_fg = mask.astype(np.float32) * scores[i]
        prob_bg = 1.0 - prob_fg
        unary = np.stack([prob_bg, prob_fg], axis=0).reshape(2, -1)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(-np.log(unary + 1e-8))
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)
        Q = d.inference(5)
        refined.append(np.argmax(Q, axis=0).reshape(H, W).astype(np.uint8))
    return np.array(refined)
