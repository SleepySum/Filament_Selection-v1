import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.dataset import SolarFilamentDataset, get_dataloader, create_train_val_split
from src.models import FilamentUNet, build_model
from src.losses import TverskyFocalLoss, ComboLoss, create_loss
from src.augmentations import get_train_augmentations, get_val_augmentations
from src.evaluator import FilamentEvaluator, run_local_validation
from src.postprocess import postprocess_batch, postprocess_masks


def parse_args():
    parser = argparse.ArgumentParser(description='Train Solar Filament Segmentation Model')
    parser.add_argument('--data_dir', type=str, default='filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026', help='Data directory')
    parser.add_argument('--json_path', type=str, default='filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json', help='Annotations JSON')
    parser.add_argument('--images_dir', type=str, default='filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images', help='Images directory')
    parser.add_argument('--model_type', type=str, default='unet', help='Model type (unet, mask2former, solov2)')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder backbone for UNet')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--loss', type=str, default='tversky_focal', help='Loss function')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """Build optimizer."""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, epochs: int):
    """Build learning rate scheduler."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler = None,
    mixed_precision: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, targets in pbar:
        images = images.to(device)
        
        # Prepare targets
        # For UNet: combine instance masks into single mask
        batch_masks = []
        for target in targets:
            masks = target['masks']  # (N, H, W)
            if masks.shape[0] > 0:
                combined = masks.sum(dim=0).clamp(0, 1)
            else:
                combined = torch.zeros_like(masks[0:1]).squeeze(0)
            batch_masks.append(combined)
        
        target_masks = torch.stack(batch_masks).to(device)  # (B, H, W)
        target_masks = target_masks.unsqueeze(1)  # (B, 1, H, W)
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler:
            with autocast():
                instance_logits, semantic_logits = model(images)
                loss = loss_fn(instance_logits, target_masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            instance_logits, semantic_logits = model(images)
            loss = loss_fn(instance_logits, target_masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    evaluator: FilamentEvaluator,
    mixed_precision: bool = False,
) -> Dict:
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation', leave=False):
            images = images.to(device)
            
            # Prepare targets
            batch_masks = []
            for target in targets:
                masks = target['masks']
                if masks.shape[0] > 0:
                    combined = masks.sum(dim=0).clamp(0, 1)
                else:
                    combined = torch.zeros_like(masks[0:1]).squeeze(0)
                batch_masks.append(combined)
            
            target_masks = torch.stack(batch_masks).to(device)
            target_masks = target_masks.unsqueeze(1)
            
            if mixed_precision:
                with autocast():
                    instance_logits, semantic_logits = model(images)
                    loss = loss_fn(instance_logits, target_masks)
            else:
                instance_logits, semantic_logits = model(images)
                loss = loss_fn(instance_logits, target_masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions for evaluation
            instance_probs = torch.sigmoid(instance_logits)
            semantic_probs = torch.sigmoid(semantic_logits)
            combined = instance_probs * semantic_probs
            
            for b in range(images.shape[0]):
                pred_masks = combined[b].cpu().numpy()
                pred_masks = (pred_masks > 0.5).astype(np.uint8)
                
                # Filter by size
                pred_list = []
                for i in range(pred_masks.shape[0]):
                    if pred_masks[i].sum() >= 150:
                        pred_list.append(pred_masks[i])
                
                gt_masks = targets[b]['masks'].cpu().numpy()
                gt_list = [gt_masks[i] for i in range(gt_masks.shape[0]) if gt_masks[i].sum() >= 150]
                
                all_predictions.append(pred_list)
                all_ground_truths.append(gt_list)
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(all_predictions, all_ground_truths)
    metrics['val_loss'] = total_loss / num_batches
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    path: str,
):
    """Save checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create train/val split
    print("Creating train/val split...")
    train_infos, val_infos = create_train_val_split(args.json_path, args.val_ratio, args.seed)
    print(f"Train samples: {len(train_infos)}, Val samples: {len(val_infos)}")
    
    # Create datasets
    train_transform = get_train_augmentations()
    val_transform = get_val_augmentations()
    
    train_dataset = SolarFilamentDataset(
        json_path=args.json_path,
        images_dir=args.images_dir,
        transform=train_transform,
        mode='train',
    )
    train_dataset.image_infos = train_infos
    
    val_dataset = SolarFilamentDataset(
        json_path=args.json_path,
        images_dir=args.images_dir,
        transform=val_transform,
        mode='val',
    )
    val_dataset.image_infos = val_infos
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: (torch.stack([i['image'] for i in x]), [i['target'] for i in x]),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: (torch.stack([i['image'] for i in x]), [i['target'] for i in x]),
        pin_memory=True,
    )
    
    # Build model
    print(f"Building {args.model_type} model...")
    if args.model_type == 'unet':
        model = FilamentUNet(encoder=args.encoder, num_classes=1, num_instances=32)
    else:
        model = build_model(args.model_type)
    
    model.to(device)
    
    # Loss
    loss_fn = create_loss(args.loss)
    
    # Optimizer and scheduler
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs)
    
    # Mixed precision
    scaler = GradScaler() if args.mixed_precision else None
    
    # Evaluator
    evaluator = FilamentEvaluator(iou_threshold=0.5)
    
    # Resume
    start_epoch = 0
    best_pq = 0
    
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pq = checkpoint['metrics'].get('PQ', 0)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, args.mixed_precision
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, evaluator, args.mixed_precision)
        
        # Scheduler step
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        evaluator.print_summary(val_metrics)
        
        # Save best model
        if val_metrics['PQ'] > best_pq:
            best_pq = val_metrics['PQ']
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path)
            print(f"Saved best model (PQ: {best_pq:.4f}) to {save_path}")
        
        # Save latest
        save_path = os.path.join(args.output_dir, 'latest_model.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path)
    
    print(f"\nTraining complete! Best PQ: {best_pq:.4f}")


if __name__ == '__main__':
    main()