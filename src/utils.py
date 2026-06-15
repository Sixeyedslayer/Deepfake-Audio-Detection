"""
utils.py — Helper functions for the deepfake audio detection project.
Includes EER computation, logging setup, reproducibility seeding, and device selection.
"""

import os
import random
import logging
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("  Using CPU (training will be slow)")
    return device


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: Ground truth labels (1 = genuine, 0 = fake)
        y_scores: Model prediction scores (higher = more likely genuine)
    
    Returns:
        eer: Equal Error Rate as a float (0-1)
        threshold: The threshold at which EER occurs
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # EER is where FPR = 1 - TPR (i.e., FPR = FNR)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    # Find the threshold at EER
    eer_threshold = interp1d(fpr, thresholds)(eer)
    
    return float(eer), float(eer_threshold)


def setup_logging(log_dir: str = "logs", name: str = "deepfake_detection") -> logging.Logger:
    """Set up logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.
    
    Monitors a metric and stops training if it hasn't improved for `patience` epochs.
    For EER, lower is better (mode='min'). For accuracy, higher is better (mode='max').
    """
    
    def __init__(self, patience: int = 7, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.mode == "min":
            improved = metric < (self.best_score - self.min_delta)
        else:
            improved = metric > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
