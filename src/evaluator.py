import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pycocotools import mask as mask_utils
from collections import defaultdict
import matplotlib.pyplot as plt


class FilamentEvaluator:
    """
    Evaluator for Solar Filament Segmentation Challenge 2026.
    Replicates official competition evaluator using PyTorch matrix multiplication.
    Computes Panoptic Quality (PQ) at IoU threshold 0.5.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        min_instance_area: int = 150,
    ):
        """
        Args:
            iou_threshold: IoU threshold for matching (default 0.5)
            min_instance_area: Minimum instance area in pixels
        """
        self.iou_threshold = iou_threshold
        self.min_instance_area = min_instance_area
        
        # Metrics storage
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.sum_iou = 0.0
        
        # For diagnostic distributions
        self.iou_scores = []
        self.dice_scores = []
        self.fragmentation_counts = []  # 1:n links
        self.overmerging_counts = []    # m:1 links
    
    def _rle_to_tensor(self, rle: dict, device: torch.device) -> torch.Tensor:
        """Convert RLE to binary tensor."""
        mask = mask_utils.decode(rle)
        return torch.from_numpy(mask).to(device).float()
    
    def _masks_to_tensor(self, masks: List[np.ndarray], device: torch.device) -> torch.Tensor:
        """Convert list of binary masks to tensor (N, H, W)."""
        if len(masks) == 0:
            return torch.zeros((0, masks[0].shape[0], masks[0].shape[1]), device=device)
        return torch.from_numpy(np.stack(masks)).to(device).float()
    
    def compute_iou_matrix(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU matrix between predicted and ground truth masks using matrix multiplication.
        
        Args:
            pred_masks: (N_pred, H, W) binary tensors
            gt_masks: (N_gt, H, W) binary tensors
            
        Returns:
            IoU matrix of shape (N_pred, N_gt)
        """
        N_pred = pred_masks.shape[0]
        N_gt = gt_masks.shape[0]
        
        if N_pred == 0 or N_gt == 0:
            return torch.zeros((N_pred, N_gt), device=pred_masks.device)
        
        # Flatten masks: (N, H*W)
        pred_flat = pred_masks.view(N_pred, -1)
        gt_flat = gt_masks.view(N_gt, -1)
        
        # Intersection: (N_pred, N_gt)
        intersection = torch.mm(pred_flat, gt_flat.t())
        
        # Areas
        pred_areas = pred_flat.sum(dim=1, keepdim=True)  # (N_pred, 1)
        gt_areas = gt_flat.sum(dim=1, keepdim=True).t()  # (1, N_gt)
        
        # Union
        union = pred_areas + gt_areas - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def match_instances(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predicted instances to ground truth using greedy IoU matching.
        
        Args:
            pred_masks: (N_pred, H, W)
            gt_masks: (N_gt, H, W)
            
        Returns:
            matched_pairs: List of (pred_idx, gt_idx)
            unmatched_pred: List of unmatched pred indices
            unmatched_gt: List of unmatched gt indices
        """
        iou_matrix = self.compute_iou_matrix(pred_masks, gt_masks)
        N_pred, N_gt = iou_matrix.shape
        
        matched_pairs = []
        matched_pred = set()
        matched_gt = set()
        
        # Flatten and sort by IoU descending
        iou_flat = iou_matrix.view(-1)
        sorted_indices = torch.argsort(iou_flat, descending=True)
        
        for idx in sorted_indices:
            pred_idx = idx // N_gt
            gt_idx = idx % N_gt
            iou_val = iou_flat[idx].item()
            
            if iou_val < self.iou_threshold:
                break
            
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matched_pairs.append((pred_idx.item(), gt_idx.item()))
                matched_pred.add(pred_idx.item())
                matched_gt.add(gt_idx.item())
        
        unmatched_pred = [i for i in range(N_pred) if i not in matched_pred]
        unmatched_gt = [i for i in range(N_gt) if i not in matched_gt]
        
        return matched_pairs, unmatched_pred, unmatched_gt
    
    def evaluate_sample(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate a single sample (image).
        
        Args:
            pred_masks: List of predicted binary masks
            gt_masks: List of ground truth binary masks
            
        Returns:
            Dict with PQ, TP, FP, FN, and diagnostic info
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Filter by minimum area
        pred_masks = [m for m in pred_masks if m.sum() >= self.min_instance_area]
        gt_masks = [m for m in gt_masks if m.sum() >= self.min_instance_area]
        
        if len(pred_masks) == 0 and len(gt_masks) == 0:
            return {
                'pq': 1.0, 'tp': 0, 'fp': 0, 'fn': 0,
                'sum_iou': 0.0, 'num_pred': 0, 'num_gt': 0
            }
        
        # Convert to tensors
        pred_tensor = self._masks_to_tensor(pred_masks, device)
        gt_tensor = self._masks_to_tensor(gt_masks, device)
        
        # Match instances
        matched_pairs, unmatched_pred, unmatched_gt = self.match_instances(pred_tensor, gt_tensor)
        
        # Compute metrics
        tp = len(matched_pairs)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        # Sum IoU for matched pairs
        sum_iou = 0.0
        for pred_idx, gt_idx in matched_pairs:
            iou_matrix = self.compute_iou_matrix(
                pred_tensor[pred_idx:pred_idx+1],
                gt_tensor[gt_idx:gt_idx+1]
            )
            iou_val = iou_matrix[0, 0].item()
            sum_iou += iou_val
            self.iou_scores.append(iou_val)
            
            # Also compute Dice
            pred_mask = pred_tensor[pred_idx]
            gt_mask = gt_tensor[gt_idx]
            intersection = (pred_mask * gt_mask).sum()
            dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
            self.dice_scores.append(dice.item())
        
        # Compute fragmentation (1:n) and over-merging (m:1)
        # For each GT, count how many preds match it
        gt_to_preds = defaultdict(list)
        for pred_idx, gt_idx in matched_pairs:
            gt_to_preds[gt_idx].append(pred_idx)
        
        fragmentation = 0
        overmerging = 0
        for gt_idx, pred_list in gt_to_preds.items():
            if len(pred_list) > 1:
                fragmentation += len(pred_list) - 1  # 1:n penalty
        
        # For each pred, count how many GTs it matches (should be 1 in greedy matching)
        # But check for over-merging by looking at IoU with all GTs
        for pred_idx in range(pred_tensor.shape[0]):
            if pred_idx in unmatched_pred:
                continue
            # Check if this pred has high IoU with multiple GTs
            iou_row = self.compute_iou_matrix(
                pred_tensor[pred_idx:pred_idx+1],
                gt_tensor
            )[0]
            high_iou_gts = (iou_row > self.iou_threshold).sum().item()
            if high_iou_gts > 1:
                overmerging += high_iou_gts - 1
        
        self.fragmentation_counts.append(fragmentation)
        self.overmerging_counts.append(overmerging)
        
        # Panoptic Quality
        pq = sum_iou / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
        
        return {
            'pq': pq,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'sum_iou': sum_iou,
            'num_pred': len(pred_masks),
            'num_gt': len(gt_masks),
            'fragmentation': fragmentation,
            'overmerging': overmerging,
        }
    
    def evaluate_dataset(
        self,
        predictions: List[List[np.ndarray]],
        ground_truths: List[List[np.ndarray]]
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset.
        
        Args:
            predictions: List of lists of predicted masks per image
            ground_truths: List of lists of ground truth masks per image
            
        Returns:
            Aggregate metrics dict
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.sum_iou = 0.0
        self.iou_scores = []
        self.dice_scores = []
        self.fragmentation_counts = []
        self.overmerging_counts = []
        
        for pred_masks, gt_masks in zip(predictions, ground_truths):
            result = self.evaluate_sample(pred_masks, gt_masks)
            self.tp += result['tp']
            self.fp += result['fp']
            self.fn += result['fn']
            self.sum_iou += result['sum_iou']
        
        # Final PQ
        pq = self.sum_iou / (self.tp + 0.5 * self.fp + 0.5 * self.fn) \
            if (self.tp + 0.5 * self.fp + 0.5 * self.fn) > 0 else 0.0
        
        return {
            'PQ': pq,
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn,
            'Sum_IoU': self.sum_iou,
            'Mean_IoU': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'Mean_Dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'Total_Fragmentation': sum(self.fragmentation_counts),
            'Total_Overmerging': sum(self.overmerging_counts),
            'Fragmentation_Rate': sum(self.fragmentation_counts) / max(self.tp, 1),
            'Overmerging_Rate': sum(self.overmerging_counts) / max(self.tp, 1),
        }
    
    def plot_diagnostics(self, save_path: Optional[str] = None):
        """Plot diagnostic distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # IoU histogram
        axes[0, 0].hist(self.iou_scores, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.iou_threshold, color='red', linestyle='--', label=f'Threshold ({self.iou_threshold})')
        axes[0, 0].set_xlabel('IoU')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('IoU Distribution (Matched Pairs)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice histogram
        axes[0, 1].hist(self.dice_scores, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Dice Coefficient')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Dice Distribution (Matched Pairs)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fragmentation
        axes[1, 0].hist(self.fragmentation_counts, bins=range(max(self.fragmentation_counts)+2), 
                       edgecolor='black', alpha=0.7, align='left')
        axes[1, 0].set_xlabel('Fragmentation Count (1:n)')
        axes[1, 0].set_ylabel('Number of Images')
        axes[1, 0].set_title('Fragmentation Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Over-merging
        axes[1, 1].hist(self.overmerging_counts, bins=range(max(self.overmerging_counts)+2),
                       edgecolor='black', alpha=0.7, align='left')
        axes[1, 1].set_xlabel('Over-merging Count (m:1)')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Over-merging Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_summary(self, metrics: Dict[str, float]):
        """Print evaluation summary."""
        print("=" * 60)
        print("SOLAR FILAMENT SEGMENTATION - EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Panoptic Quality (PQ):     {metrics['PQ']:.4f}")
        print(f"True Positives (TP):       {metrics['TP']}")
        print(f"False Positives (FP):      {metrics['FP']}")
        print(f"False Negatives (FN):      {metrics['FN']}")
        print(f"Sum IoU:                   {metrics['Sum_IoU']:.4f}")
        print(f"Mean IoU (matched):        {metrics['Mean_IoU']:.4f}")
        print(f"Mean Dice (matched):       {metrics['Mean_Dice']:.4f}")
        print(f"Total Fragmentation (1:n): {metrics['Total_Fragmentation']}")
        print(f"Total Over-merging (m:1):  {metrics['Total_Overmerging']}")
        print(f"Fragmentation Rate:        {metrics['Fragmentation_Rate']:.4f}")
        print(f"Over-merging Rate:         {metrics['Overmerging_Rate']:.4f}")
        print("=" * 60)


def evaluate_predictions(
    pred_rles: List[dict],
    gt_rles: List[dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Convenience function to evaluate predictions in RLE format.
    
    Args:
        pred_rles: List of predicted RLE dicts
        gt_rles: List of ground truth RLE dicts
        iou_threshold: IoU threshold
        
    Returns:
        Metrics dict
    """
    evaluator = FilamentEvaluator(iou_threshold=iou_threshold)
    
    pred_masks = [mask_utils.decode(rle) for rle in pred_rles]
    gt_masks = [mask_utils.decode(rle) for rle in gt_rles]
    
    return evaluator.evaluate_dataset([pred_masks], [gt_masks])


# Local validation harness
def run_local_validation(
    model,
    dataloader,
    device: torch.device,
    postprocess_fn=None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run local validation on a dataloader.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Torch device
        postprocess_fn: Optional postprocessing function
        iou_threshold: IoU threshold for PQ
        
    Returns:
        Validation metrics
    """
    model.eval()
    evaluator = FilamentEvaluator(iou_threshold=iou_threshold)
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # Model inference
            outputs = model(images)
            
            # Postprocess
            if postprocess_fn:
                pred_masks_batch = postprocess_fn(outputs)
            else:
                # Default: threshold at 0.5
                pred_masks_batch = []
                for out in outputs:
                    if 'masks' in out:
                        masks = out['masks'].cpu().numpy()
                        masks = (masks > 0.5).astype(np.uint8)
                        pred_masks_batch.append([m for m in masks])
                    else:
                        pred_masks_batch.append([])
            
            # Ground truth
            gt_masks_batch = []
            for target in targets:
                gt_masks = target['masks'].cpu().numpy()
                gt_masks_batch.append([m for m in gt_masks])
            
            all_predictions.extend(pred_masks_batch)
            all_ground_truths.extend(gt_masks_batch)
    
    metrics = evaluator.evaluate_dataset(all_predictions, all_ground_truths)
    evaluator.print_summary(metrics)
    
    return metrics