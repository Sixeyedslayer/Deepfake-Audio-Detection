"""
evaluate.py — Evaluation module for the deepfake audio detection model.

Computes all required metrics:
  - Overall Accuracy (≥ 80%)
  - Equal Error Rate (≤ 12%)
  - F1 Score (≥ 80%)
  - Per-class Accuracy (≥ 75% each)
  - Confusion Matrix
  - ROC Curve + AUC
  - DET Curve

Generates plots and saves a JSON report.

Usage:
    python -m src.evaluate --model models/best_model.pth --data_dir data/for-norm
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DeepfakeDetector, build_model
from src.dataset import create_dataloaders, DeepfakeAudioDataset
from src.utils import compute_eer, get_device, set_seed


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Run model inference on a DataLoader and collect predictions.
    
    Returns:
        labels: numpy array of ground truth labels
        scores: numpy array of model prediction scores (P(genuine))
    """
    model.eval()
    all_labels = []
    all_scores = []
    
    for features, labels in tqdm(loader, desc="  Evaluating", ncols=100):
        features = features.to(device, non_blocking=True)
        
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(features)
        
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy().flatten())
    
    return np.array(all_labels), np.array(all_scores)


def compute_all_metrics(labels, scores, threshold=0.5):
    """
    Compute all required evaluation metrics.
    
    Args:
        labels: Ground truth (1=genuine, 0=fake)
        scores: Model scores (P(genuine))
        threshold: Classification threshold
    
    Returns:
        Dictionary of all metrics
    """
    preds = (scores >= threshold).astype(int)
    
    # Overall metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    
    # Per-class accuracy
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # EER
    eer, eer_threshold = compute_eer(labels, scores)
    
    # ROC AUC
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0
    
    metrics = {
        "overall_accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "roc_auc": float(auc),
        "per_class_accuracy": {
            "fake": float(per_class_acc[0]),
            "genuine": float(per_class_acc[1]),
        },
        "confusion_matrix": cm.tolist(),
        "classification_threshold": float(threshold),
        "total_samples": int(len(labels)),
        "genuine_samples": int(labels.sum()),
        "fake_samples": int(len(labels) - labels.sum()),
    }
    
    return metrics


def check_thresholds(metrics):
    """Check if all required thresholds are met."""
    checks = {
        "EER <= 10%": metrics["eer"] <= 0.10,
        "ROC AUC >= 0.90": metrics["roc_auc"] >= 0.90,
        "Accuracy >= 80%": metrics["overall_accuracy"] >= 0.80,
        "Per-class Acc (Fake) >= 75%": metrics["per_class_accuracy"]["fake"] >= 0.75,
        "Per-class Acc (Genuine) >= 75%": metrics["per_class_accuracy"]["genuine"] >= 0.75,
    }
    
    print("\n  +" + "-" * 45 + "+")
    print("  |         THRESHOLD VERIFICATION              |")
    print("  +" + "-" * 45 + "+")
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        icon = "[OK]" if passed else "[XX]"
        print(f"  | {icon} {check_name:<30s} {status:>8s} |")
        if not passed:
            all_passed = False
    
    print("  +" + "-" * 45 + "+")
    if all_passed:
        print("  |       [+] ALL THRESHOLDS MET                |")
    else:
        print("  |       [-] SOME THRESHOLDS NOT MET           |")
    print("  +" + "-" * 45 + "+")
    
    return all_passed


def plot_confusion_matrix(cm, save_path):
    """Generate and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Deepfake", "Genuine"],
        yticklabels=["Deepfake", "Genuine"],
        annot_kws={"size": 16}
    )
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix — Deepfake Audio Detection", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curve(labels, scores, save_path):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2563eb", linewidth=2.5,
             label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--",
             label="Random Classifier")
    
    # Mark EER point
    eer, _ = compute_eer(labels, scores)
    plt.plot(eer, 1 - eer, "ro", markersize=10, label=f"EER = {eer:.4f}")
    
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve — Deepfake Audio Detection", fontsize=15)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_det_curve(labels, scores, save_path):
    """Generate and save Detection Error Tradeoff (DET) curve."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr * 100, fnr * 100, color="#dc2626", linewidth=2.5,
             label="DET Curve")
    plt.plot([0, 100], [0, 100], color="gray", linewidth=1, linestyle="--")
    
    # Mark EER point
    eer, _ = compute_eer(labels, scores)
    plt.plot(eer * 100, eer * 100, "bo", markersize=10,
             label=f"EER = {eer*100:.2f}%")
    
    plt.xlabel("False Acceptance Rate (%)", fontsize=13)
    plt.ylabel("False Rejection Rate (%)", fontsize=13)
    plt.title("DET Curve — Deepfake Audio Detection", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_score_distribution(labels, scores, save_path):
    """Plot distribution of scores for genuine vs fake samples."""
    plt.figure(figsize=(10, 6))
    
    genuine_scores = scores[labels == 1]
    fake_scores = scores[labels == 0]
    
    plt.hist(genuine_scores, bins=50, alpha=0.6, color="#16a34a",
             label=f"Genuine (n={len(genuine_scores)})", density=True)
    plt.hist(fake_scores, bins=50, alpha=0.6, color="#dc2626",
             label=f"Deepfake (n={len(fake_scores)})", density=True)
    
    plt.axvline(x=0.5, color="black", linewidth=1.5, linestyle="--",
                label="Threshold = 0.5")
    
    plt.xlabel("Model Score (P(Genuine))", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.title("Score Distribution — Genuine vs Deepfake", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_metrics_summary(metrics):
    """Print a formatted summary of all metrics."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Samples:      {metrics['total_samples']:,}")
    print(f"  Genuine Samples:    {metrics['genuine_samples']:,}")
    print(f"  Deepfake Samples:   {metrics['fake_samples']:,}")
    print("  " + "-" * 40)
    print(f"  Overall Accuracy:   {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"  F1 Score:           {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
    print("  " + "-" * 40)
    print(f"  EER:                {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"  EER Threshold:      {metrics['eer_threshold']:.4f}")
    print("  " + "-" * 40)
    print(f"  Per-class Acc:")
    print(f"    Deepfake:         {metrics['per_class_accuracy']['fake']:.4f} ({metrics['per_class_accuracy']['fake']*100:.2f}%)")
    print(f"    Genuine:          {metrics['per_class_accuracy']['genuine']:.4f} ({metrics['per_class_accuracy']['genuine']*100:.2f}%)")
    print("  " + "-" * 40)
    print(f"  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                     Pred Fake  Pred Genuine")
    print(f"    True Fake:       {cm[0][0]:>8d}    {cm[0][1]:>8d}")
    print(f"    True Genuine:    {cm[1][0]:>8d}    {cm[1][1]:>8d}")
    print("=" * 60)


def evaluate(args):
    """Main evaluation function."""
    set_seed(42)
    device = get_device()
    
    print("\n" + "=" * 60)
    print("  DEEPFAKE AUDIO DETECTION — EVALUATION")
    print("=" * 60)
    
    # Create reports directory
    os.makedirs(args.report_dir, exist_ok=True)
    
    # ─── Load Model ──────────────────────────────────────────────────
    print("\n[1/3] Loading model...")
    model = build_model(device=device, pretrained=False)
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint val EER: {checkpoint.get('val_eer', 'N/A')}")
    
    # ─── Load Data ───────────────────────────────────────────────────
    print("\n[2/3] Loading test data...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_loader = dataloaders["test"]
    
    # ─── Run Evaluation ──────────────────────────────────────────────
    print("\n[3/3] Running evaluation...")
    labels, scores = get_predictions(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_all_metrics(labels, scores)
    
    # Print results
    print_metrics_summary(metrics)
    
    # Check thresholds
    check_thresholds(metrics)
    
    # ─── Generate Plots ──────────────────────────────────────────────
    print("\n  Generating plots...")
    cm = np.array(metrics["confusion_matrix"])
    
    plot_confusion_matrix(cm, os.path.join(args.report_dir, "confusion_matrix.png"))
    plot_roc_curve(labels, scores, os.path.join(args.report_dir, "roc_curve.png"))
    plot_det_curve(labels, scores, os.path.join(args.report_dir, "det_curve.png"))
    plot_score_distribution(labels, scores, os.path.join(args.report_dir, "score_distribution.png"))
    
    # ─── Save Report ─────────────────────────────────────────────────
    report_path = os.path.join(args.report_dir, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to: {report_path}")
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Audio Detector")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/for-norm",
                        help="Path to the for-norm dataset directory")
    parser.add_argument("--report_dir", type=str, default="reports",
                        help="Directory to save evaluation reports")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
