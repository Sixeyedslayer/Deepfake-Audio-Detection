"""
train.py — Training pipeline for the deepfake audio detection model.

Features:
  - Mixed precision training (FP16) for RTX 3050
  - Cosine annealing learning rate scheduler with warmup
  - Early stopping by validation EER
  - Model checkpointing (best + last)
  - Comprehensive logging

Usage:
    python -m src.train
    python -m src.train --epochs 30 --batch_size 16 --data_dir data/for-norm
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model
from src.dataset import create_dataloaders
from src.utils import (
    set_seed, get_device, compute_eer,
    AverageMeter, EarlyStopping, setup_logging
)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter("loss")
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False, ncols=100)
    
    for features, labels in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(features)
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        loss_meter.update(loss.item(), features.size(0))
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{correct/total:.3f}")
    
    return loss_meter.avg, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and compute metrics."""
    model.eval()
    loss_meter = AverageMeter("val_loss")
    
    all_labels = []
    all_scores = []
    
    for features, labels in tqdm(loader, desc="  Validate", leave=False, ncols=100):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(features)
            loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), features.size(0))
        
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        all_scores.extend(scores)
        all_labels.extend(labels.cpu().numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Compute metrics
    preds = (all_scores >= 0.5).astype(float)
    accuracy = np.mean(preds == all_labels)
    
    try:
        eer, eer_threshold = compute_eer(all_labels, all_scores)
    except Exception:
        eer, eer_threshold = 0.5, 0.5
    
    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = get_device()
    logger = setup_logging("logs")
    
    print("\n" + "=" * 60)
    print("  DEEPFAKE AUDIO DETECTION — TRAINING")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # ─── Data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples
    )
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    train_dataset = dataloaders["train_dataset"]
    
    # ─── Model ───────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model(device=device, pretrained=True)
    
    # ─── Loss, Optimizer, Scheduler ──────────────────────────────────
    print("\n[3/4] Setting up training...")
    
    # Class-weighted loss
    pos_weight = train_dataset.get_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.3f})")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    
    # Cosine annealing with warmup
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs - warmup_epochs), T_mult=1
    )
    
    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
    
    # ─── Training Loop ───────────────────────────────────────────────
    print(f"\n[4/4] Training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print()
    
    best_eer = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_eer": [], "lr": []}
    
    start_epoch = 1
    
    # ─── Resume Checkpoint ───────────────────────────────────────────
    if getattr(args, "resume", False):
        last_ckpt = os.path.join(args.model_dir, "last_model.pth")
        if os.path.exists(last_ckpt):
            print(f"  [!] Resuming from checkpoint: {last_ckpt}")
            checkpoint = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            
            best_ckpt = os.path.join(args.model_dir, "best_model.pth")
            if os.path.exists(best_ckpt):
                best_data = torch.load(best_ckpt, map_location=device, weights_only=False)
                best_eer = best_data.get("val_eer", float("inf"))
                best_epoch = best_data.get("epoch", 0)
            
            hist_path = os.path.join(args.model_dir, "training_history.json")
            if os.path.exists(hist_path):
                with open(hist_path, "r") as f:
                    history = json.load(f)
        else:
            print(f"  [!] --resume specified, but no checkpoint found at {last_ckpt}. Starting from scratch.")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Warmup: linearly increase LR for first few epochs
        if epoch <= warmup_epochs:
            warmup_lr = args.lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
            current_lr = warmup_lr
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler (after warmup)
        if epoch > warmup_epochs:
            scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_eer"].append(val_metrics["eer"])
        history["lr"].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} "
              f"EER: {val_metrics['eer']:.4f} | "
              f"LR: {current_lr:.6f} | {epoch_time:.1f}s")
        
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}, "
            f"val_eer={val_metrics['eer']:.4f}"
        )
        
        # Save best model
        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_eer": val_metrics["eer"],
                "val_accuracy": val_metrics["accuracy"],
                "eer_threshold": val_metrics["eer_threshold"],
                "args": vars(args),
            }, os.path.join(args.model_dir, "best_model.pth"))
            print(f"  [*] New best model saved (EER: {best_eer:.4f})")
        
        # Save last model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_eer": val_metrics["eer"],
            "val_accuracy": val_metrics["accuracy"],
            "args": vars(args),
        }, os.path.join(args.model_dir, "last_model.pth"))
        
        # Early stopping
        if early_stopping(val_metrics["eer"]):
            print(f"\n  Early stopping triggered at epoch {epoch}")
            print(f"  Best EER: {best_eer:.4f} at epoch {best_epoch}")
            break
    
    # ─── Training Complete ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best EER: {best_eer:.4f} at epoch {best_epoch}")
    print(f"  Model saved to: {os.path.join(args.model_dir, 'best_model.pth')}")
    
    # Save training history
    history_path = os.path.join(args.model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to: {history_path}")
    
    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Audio Detector")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/for-norm",
                        help="Path to the for-norm dataset directory")
    parser.add_argument("--cache_dir", type=str, default="features_cache",
                        help="Directory to cache extracted features")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split (for debugging)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from models/last_model.pth")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (16 for RTX 3050 4GB)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (2 for Windows stability)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
