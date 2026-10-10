import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    Focuses on thin filament barbs with alpha=0.3, beta=0.7.
    
    Tversky Index = TP / (TP + alpha*FP + beta*FN)
    Tversky Loss = 1 - Tversky Index
    
    Args:
        alpha: Weight for false positives (default 0.3)
        beta: Weight for false negatives (default 0.7)
        smooth: Smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        from_logits: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits or probabilities
            targets: (B, C, H, W) binary masks
            
        Returns:
            Tversky loss scalar
        """
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL = -alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        alpha: Weighting factor for rare class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, C, H, W) binary masks
            
        Returns:
            Focal loss
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)
        
        # Compute pt
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = focal_weight * (1 - pt).pow(self.gamma)
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TverskyFocalLoss(nn.Module):
    """
    Combined Tversky + Focal Loss for filament segmentation.
    Tversky focuses on thin structures (barbs), Focal handles class imbalance.
    """
    
    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_weight: float = 0.7,
        focal_weight: float = 0.3,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
            from_logits=True,
        )
        self.focal = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
        )
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, C, H, W) binary masks
            
        Returns:
            Combined loss
        """
        tversky_loss = self.tversky(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        
        return self.tversky_weight * tversky_loss + self.focal_weight * focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class ComboLoss(nn.Module):
    """
    Combination of multiple losses for robust training.
    """
    
    def __init__(
        self,
        weights: dict = None,
    ):
        super().__init__()
        self.weights = weights or {
            'bce': 1.0,
            'dice': 1.0,
            'tversky': 1.0,
            'focal': 0.5,
        }
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()
        self.focal = FocalLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        
        if self.weights.get('bce', 0) > 0:
            loss += self.weights['bce'] * self.bce(inputs, targets)
        
        if self.weights.get('dice', 0) > 0:
            loss += self.weights['dice'] * self.dice(inputs, targets)
        
        if self.weights.get('tversky', 0) > 0:
            loss += self.weights['tversky'] * self.tversky(inputs, targets)
        
        if self.weights.get('focal', 0) > 0:
            loss += self.weights['focal'] * self.focal(inputs, targets)
        
        return loss


class InstanceSegmentationLoss(nn.Module):
    """
    Loss for instance segmentation with mask and category heads.
    """
    
    def __init__(
        self,
        mask_loss: nn.Module = None,
        cate_loss: nn.Module = None,
        mask_weight: float = 1.0,
        cate_weight: float = 1.0,
    ):
        super().__init__()
        self.mask_loss = mask_loss or TverskyFocalLoss()
        self.cate_loss = cate_loss or nn.BCEWithLogitsLoss()
        self.mask_weight = mask_weight
        self.cate_weight = cate_weight
    
    def forward(
        self,
        mask_preds: List[torch.Tensor],
        cate_preds: List[torch.Tensor],
        mask_targets: List[torch.Tensor],
        cate_targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mask_preds: List of mask predictions per FPN level
            cate_preds: List of category predictions per FPN level
            mask_targets: List of mask targets per FPN level
            cate_targets: List of category targets per FPN level
            
        Returns:
            Dict of losses
        """
        total_mask_loss = 0
        total_cate_loss = 0
        
        for mask_pred, mask_target in zip(mask_preds, mask_targets):
            total_mask_loss += self.mask_loss(mask_pred, mask_target)
        
        for cate_pred, cate_target in zip(cate_preds, cate_targets):
            total_cate_loss += self.cate_loss(cate_pred, cate_target)
        
        total_loss = self.mask_weight * total_mask_loss + self.cate_weight * total_cate_loss
        
        return {
            'loss': total_loss,
            'mask_loss': total_mask_loss,
            'cate_loss': total_cate_loss,
        }


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Loss arguments
        
    Returns:
        Loss module
    """
    losses = {
        'tversky': TverskyLoss,
        'focal': FocalLoss,
        'tversky_focal': TverskyFocalLoss,
        'dice': DiceLoss,
        'combo': ComboLoss,
        'instance': InstanceSegmentationLoss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}")
    
    return losses[loss_name](**kwargs)