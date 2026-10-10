import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import warnings
warnings.filterwarnings('ignore')


class Mask2FormerFilament(nn.Module):
    """
    Mask2Former for solar filament instance segmentation.
    Uses Swin-L backbone for high-resolution feature extraction.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 100,
        backbone: str = 'swin_large_patch4_window12_384',
        hidden_dim: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        mask_dim: int = 256,
        enforce_input_project: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Build Mask2Former using detectron2
        self.cfg = self._setup_cfg()
        self.model = build_model(self.cfg)
        
        # Replace classifier head for single class
        self._replace_heads(num_classes)
        
    def _setup_cfg(self):
        """Setup detectron2 config for Mask2Former."""
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.projects.mask2former import add_mask2former_config
        
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_mask2former_config(cfg)
        
        # Load Mask2Former config
        cfg.merge_from_file(
            "detectron2/projects/Mask2Former/configs/coco/instance-segmentation/mask2former_R50_bs16_50ep.yaml"
        )
        
        # Modify for our task
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = self.num_queries
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.num_classes
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.5
        cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.5
        
        # Input size
        cfg.INPUT.MIN_SIZE_TRAIN = (2048,)
        cfg.INPUT.MAX_SIZE_TRAIN = 2048
        cfg.INPUT.MIN_SIZE_TEST = 2048
        cfg.INPUT.MAX_SIZE_TEST = 2048
        
        return cfg
    
    def _replace_heads(self, num_classes: int):
        """Replace classification heads for single class."""
        # This would need detectron2 specific modifications
        pass
    
    def forward(self, images: torch.Tensor) -> List[Dict]:
        """
        Forward pass.
        
        Args:
            images: (B, 3, H, W) tensor
            
        Returns:
            List of dicts with 'masks', 'scores', 'labels'
        """
        # Prepare inputs for detectron2
        inputs = []
        for i in range(images.shape[0]):
            inputs.append({
                "image": images[i].permute(1, 2, 0).contiguous(),
                "height": images.shape[2],
                "width": images.shape[3],
            })
        
        # Inference
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs


class SOLOv2Filament(nn.Module):
    """
    SOLOv2 for solar filament instance segmentation.
    Simple, fast, and effective for instance segmentation.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = 'resnet50',
        fpn_strides: List[int] = [8, 16, 32, 64, 128],
        num_grids: List[int] = [40, 36, 24, 16, 12],
        scale_ranges: List[Tuple[int, int]] = [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        sigma: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        
        # Backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        
        # FPN
        self.fpn = self._build_fpn()
        
        # Category and mask kernels
        self.cate_head = self._build_head(256, num_classes)
        self.mask_head = self._build_head(256, 256)
        
        # Mask feature
        self.mask_feature = nn.Conv2d(256, 256, 3, padding=1)
        
    def _build_fpn(self) -> nn.Module:
        """Build Feature Pyramid Network."""
        class FPN(nn.Module):
            def __init__(self, in_channels_list, out_channels):
                super().__init__()
                self.lateral_convs = nn.ModuleList()
                self.fpn_convs = nn.ModuleList()
                
                for in_channels in in_channels_list:
                    self.lateral_convs.append(
                        nn.Conv2d(in_channels, out_channels, 1)
                    )
                    self.fpn_convs.append(
                        nn.Conv2d(out_channels, out_channels, 3, padding=1)
                    )
                
            def forward(self, features):
                # features: list of (B, C, H, W) from backbone
                laterals = [l(c) for l, c in zip(self.lateral_convs, features)]
                
                # Top-down
                for i in range(len(laterals) - 1, 0, -1):
                    laterals[i - 1] += F.interpolate(
                        laterals[i], scale_factor=2, mode='nearest'
                    )
                
                # FPN convs
                fpn_out = [conv(l) for conv, l in zip(self.fpn_convs, laterals)]
                return fpn_out
        
        # Get backbone feature channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            in_channels = [f.shape[1] for f in feats]
        
        return FPN(in_channels, 256)
    
    def _build_head(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build prediction head."""
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            layers.append(nn.GroupNorm(32, in_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W)
            
        Returns:
            cate_preds: List of (B, num_classes, H, W) per level
            mask_preds: List of (B, num_grids^2, H, W) per level
        """
        # Backbone
        feats = self.backbone(x)
        
        # FPN
        fpn_feats = self.fpn(feats)
        
        cate_preds = []
        mask_preds = []
        
        for i, feat in enumerate(fpn_feats):
            # Category prediction
            cate_pred = self.cate_head(feat)
            cate_preds.append(cate_pred)
            
            # Mask prediction
            mask_feat = self.mask_head(feat)
            mask_preds.append(mask_feat)
        
        return cate_preds, mask_preds
    
    def predict(self, x: torch.Tensor, score_thr: float = 0.5, nms_thr: float = 0.5) -> List[Dict]:
        """
        Generate instance predictions.
        
        Args:
            x: (B, 3, H, W)
            score_thr: Score threshold
            nms_thr: NMS IoU threshold
            
        Returns:
            List of predictions per image
        """
        cate_preds, mask_preds = self.forward(x)
        B = x.shape[0]
        results = []
        
        for b in range(B):
            # Process each level
            all_masks = []
            all_scores = []
            all_labels = []
            
            for i, (cate_pred, mask_pred) in enumerate(zip(cate_preds, mask_preds)):
                # Get predictions for this image
                cate = cate_pred[b].sigmoid()  # (C, H, W)
                mask = mask_pred[b]  # (256, H, W)
                
                # Get grid size
                num_grid = self.num_grids[i]
                H, W = cate.shape[1], cate.shape[2]
                
                # Reshape mask to grids
                mask = mask.permute(1, 2, 0)  # (H, W, 256)
                mask = mask.view(H, W, num_grid, num_grid)  # (H, W, G, G)
                mask = mask.permute(2, 3, 0, 1)  # (G, G, H, W)
                
                # Generate instance masks
                for gy in range(num_grid):
                    for gx in range(num_grid):
                        score = cate[0, gy, gx].item()
                        if score > score_thr:
                            inst_mask = mask[gy, gx]  # (H, W)
                            inst_mask = torch.sigmoid(inst_mask)
                            inst_mask = (inst_mask > 0.5).float()
                            
                            if inst_mask.sum() > 150:  # min area
                                all_masks.append(inst_mask)
                                all_scores.append(score)
                                all_labels.append(1)
            
            # NMS
            if all_masks:
                masks_tensor = torch.stack(all_masks)
                scores_tensor = torch.tensor(all_scores)
                
                # Simple NMS
                keep = self._mask_nms(masks_tensor, scores_tensor, nms_thr)
                masks_tensor = masks_tensor[keep]
                scores_tensor = scores_tensor[keep]
                
                results.append({
                    'masks': masks_tensor.cpu().numpy(),
                    'scores': scores_tensor.cpu().numpy(),
                    'labels': np.array(all_labels)[keep],
                })
            else:
                results.append({
                    'masks': np.zeros((0, x.shape[2], x.shape[3])),
                    'scores': np.array([]),
                    'labels': np.array([]),
                })
        
        return results
    
    def _mask_nms(self, masks: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
        """Mask NMS."""
        if len(masks) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Compute IoU matrix
        masks_flat = masks.view(len(masks), -1).float()
        inter = torch.mm(masks_flat, masks_flat.t())
        areas = masks_flat.sum(dim=1)
        union = areas.unsqueeze(1) + areas.unsqueeze(0) - inter
        iou = inter / (union + 1e-6)
        
        # NMS
        _, order = scores.sort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            iou_max = iou[i, order[1:]]
            order = order[1:][iou_max <= iou_thr]
        
        return torch.tensor(keep, dtype=torch.long)


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone for high-resolution filament segmentation.
    """
    
    def __init__(
        self,
        model_name: str = 'swin_large_patch4_window12_384',
        pretrained: bool = True,
        out_indices: Tuple[int] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        
        # Get feature channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            feats = self.model(dummy)
            self.feature_channels = [f.shape[1] for f in feats]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model(x)


def build_model(model_name: str = 'mask2former', **kwargs) -> nn.Module:
    """
    Factory function to build model.
    
    Args:
        model_name: 'mask2former', 'solov2', or 'swin'
        **kwargs: Model arguments
        
    Returns:
        Model instance
    """
    if model_name == 'mask2former':
        return Mask2FormerFilament(**kwargs)
    elif model_name == 'solov2':
        return SOLOv2Filament(**kwargs)
    elif model_name == 'swin':
        return SwinTransformerBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Lightweight U-Net with instance head for baseline
class FilamentUNet(nn.Module):
    """
    U-Net with instance segmentation head for baseline.
    """
    
    def __init__(
        self,
        encoder: str = 'resnet34',
        num_classes: int = 1,
        num_instances: int = 32,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = timm.create_model(
            encoder,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        
        # Get encoder channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            feats = self.encoder(dummy)
            enc_channels = [f.shape[1] for f in feats]
        
        # Decoder
        self.decoder = self._build_decoder(enc_channels)
        
        # Instance head
        self.instance_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_instances, 1),
        )
        
        # Semantic head
        self.semantic_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
    
    def _build_decoder(self, enc_channels: List[int]) -> nn.Module:
        """Build U-Net decoder."""
        class Decoder(nn.Module):
            def __init__(self, enc_channels):
                super().__init__()
                self.up_convs = nn.ModuleList()
                self.dec_convs = nn.ModuleList()
                
                in_ch = enc_channels[-1]
                for i in range(len(enc_channels) - 2, -1, -1):
                    out_ch = enc_channels[i]
                    self.up_convs.append(
                        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
                    )
                    self.dec_convs.append(
                        nn.Sequential(
                            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_ch, out_ch, 3, padding=1),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                        )
                    )
                    in_ch = out_ch
                
                self.final_up = nn.ConvTranspose2d(in_ch, 64, 2, stride=2)
                self.final_conv = nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, features):
                x = features[-1]
                for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_convs)):
                    x = up(x)
                    skip = features[i]
                    x = torch.cat([x, skip], dim=1)
                    x = dec(x)
                x = self.final_up(x)
                x = self.final_conv(x)
                return x
        
        return Decoder(enc_channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W)
            
        Returns:
            instance_logits: (B, num_instances, H, W)
            semantic_logits: (B, num_classes, H, W)
        """
        # Encoder
        feats = self.encoder(x)
        
        # Decoder
        dec_feat = self.decoder(feats)
        
        # Heads
        instance_logits = self.instance_head(dec_feat)
        semantic_logits = self.semantic_head(dec_feat)
        
        return instance_logits, semantic_logits
    
    def predict(self, x: torch.Tensor, score_thr: float = 0.5) -> List[Dict]:
        """Generate predictions."""
        instance_logits, semantic_logits = self.forward(x)
        B = x.shape[0]
        results = []
        
        instance_probs = torch.sigmoid(instance_logits)
        semantic_probs = torch.sigmoid(semantic_logits)
        
        for b in range(B):
            masks = []
            scores = []
            
            for i in range(instance_probs.shape[1]):
                mask = instance_probs[b, i] * semantic_probs[b, 0]
                mask = (mask > score_thr).float()
                if mask.sum() > 150:
                    masks.append(mask)
                    scores.append(mask.float().mean().item())
            
            if masks:
                masks = torch.stack(masks)
                scores = torch.tensor(scores)
                # NMS
                keep = self._mask_nms(masks, scores, 0.4)
                masks = masks[keep]
                scores = scores[keep]
                results.append({
                    'masks': masks.cpu().numpy(),
                    'scores': scores.cpu().numpy(),
                    'labels': np.ones(len(keep), dtype=np.int64),
                })
            else:
                results.append({
                    'masks': np.zeros((0, x.shape[2], x.shape[3])),
                    'scores': np.array([]),
                    'labels': np.array([]),
                })
        
        return results
    
    def _mask_nms(self, masks: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
        """Mask NMS."""
        if len(masks) == 0:
            return torch.tensor([], dtype=torch.long)
        
        masks_flat = masks.view(len(masks), -1).float()
        inter = torch.mm(masks_flat, masks_flat.t())
        areas = masks_flat.sum(dim=1)
        union = areas.unsqueeze(1) + areas.unsqueeze(0) - inter
        iou = inter / (union + 1e-6)
        
        _, order = scores.sort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            iou_max = iou[i, order[1:]]
            order = order[1:][iou_max <= iou_thr]
        
        return torch.tensor(keep, dtype=torch.long)