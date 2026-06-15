"""
dataset.py — PyTorch Dataset and DataLoader for the Fake-or-Real (FoR) dataset.

Handles:
  - Directory-based label assignment (real/ → 1, fake/ → 0)
  - On-the-fly feature extraction with optional caching
  - Data augmentation (SpecAugment, noise injection, time shift)
  - Train/validation/test DataLoader creation
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.preprocessing import (
    load_audio, preprocess_audio, extract_features,
    SAMPLE_RATE, MAX_SAMPLES
)


class AudioAugmentor:
    """
    Audio-level data augmentation applied BEFORE feature extraction.
    Only used during training.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, y: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return y
        
        augmentations = [
            self._add_noise,
            self._time_shift,
            self._change_speed,
            self._change_pitch,
        ]
        
        # Apply 1-2 random augmentations
        n_aug = random.randint(1, 2)
        chosen = random.sample(augmentations, n_aug)
        for aug in chosen:
            y = aug(y)
        
        return y
    
    def _add_noise(self, y: np.ndarray) -> np.ndarray:
        """Add Gaussian noise with random SNR."""
        noise_factor = random.uniform(0.001, 0.015)
        noise = np.random.randn(len(y)) * noise_factor
        return y + noise
    
    def _time_shift(self, y: np.ndarray) -> np.ndarray:
        """Shift audio in time by a random amount."""
        shift = random.randint(-SAMPLE_RATE // 4, SAMPLE_RATE // 4)
        return np.roll(y, shift)
    
    def _change_speed(self, y: np.ndarray) -> np.ndarray:
        """Change playback speed slightly."""
        import librosa
        speed_factor = random.uniform(0.9, 1.1)
        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
        # Ensure same length
        if len(y_stretched) < len(y):
            y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)))
        else:
            y_stretched = y_stretched[:len(y)]
        return y_stretched
    
    def _change_pitch(self, y: np.ndarray) -> np.ndarray:
        """Shift pitch slightly."""
        import librosa
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)


class SpecAugment:
    """
    SpecAugment: frequency and time masking applied AFTER feature extraction.
    Applied to the spectrogram tensor during training.
    """
    
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 20,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to a feature tensor.
        
        Args:
            spec: Feature array of shape (C, F, T)
        
        Returns:
            Augmented feature array
        """
        spec = spec.copy()
        _, n_freq, n_time = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_freq - 1))
            f0 = random.randint(0, n_freq - f)
            spec[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            spec[:, :, t0:t0 + t] = 0
        
        return spec


class DeepfakeAudioDataset(Dataset):
    """
    PyTorch Dataset for the Fake-or-Real audio dataset.
    
    Directory structure:
        root/
        ├── real/   → label 1 (genuine)
        └── fake/   → label 0 (deepfake)
    
    Args:
        root_dir: Path to the split directory (e.g., data/for-norm/training/)
        augment: Whether to apply data augmentation (True for training)
        cache_dir: Optional directory to cache extracted features
        max_samples: Maximum number of samples to load (for debugging)
    """
    
    LABEL_MAP = {"real": 1, "fake": 0}
    LABEL_NAMES = {0: "Deepfake", 1: "Genuine"}
    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
    
    def __init__(self, root_dir: str, augment: bool = False,
                 cache_dir: str = None, max_samples: int = None):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Augmentation transforms
        self.audio_augmentor = AudioAugmentor(p=0.5) if augment else None
        self.spec_augment = SpecAugment() if augment else None
        
        # Scan directories for audio files
        self.samples = []
        for label_name, label_id in self.LABEL_MAP.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                print(f"  Warning: Directory not found: {label_dir}")
                continue
            
            files = sorted([
                f for f in label_dir.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ])
            
            for f in files:
                self.samples.append((str(f), label_id))
        
        # Limit samples if requested (for debugging)
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Print dataset info
        n_genuine = sum(1 for _, l in self.samples if l == 1)
        n_fake = sum(1 for _, l in self.samples if l == 0)
        print(f"  Loaded {len(self.samples)} samples: "
              f"{n_genuine} genuine, {n_fake} fake "
              f"(from {self.root_dir})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        file_path, label = self.samples[idx]
        
        # Try loading from cache
        features = self._load_from_cache(file_path)
        
        if features is None:
            # Load and preprocess audio
            y = load_audio(file_path)
            y = preprocess_audio(y)
            
            # Apply audio augmentation (training only)
            if self.audio_augmentor:
                y = self.audio_augmentor(y)
            
            # Extract features
            features = extract_features(y)
            
            # Save to cache (without augmentation)
            if self.cache_dir and not self.augment:
                self._save_to_cache(file_path, features)
        
        # Apply SpecAugment (training only, after cache load)
        if self.spec_augment:
            features = self.spec_augment(features)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return features_tensor, label_tensor
    
    def _cache_path(self, file_path: str) -> Path:
        """Generate a unique cache path for a file."""
        p = Path(file_path)
        # Include parent dir (real/fake) to avoid collisions
        return self.cache_dir / f"{p.parent.name}_{p.stem}.npy"
    
    def _load_from_cache(self, file_path: str) -> np.ndarray:
        """Load cached features if available."""
        if self.cache_dir is None:
            return None
        cache_file = self._cache_path(file_path)
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, file_path: str, features: np.ndarray):
        """Save extracted features to cache."""
        if self.cache_dir is None:
            return
        cache_file = self._cache_path(file_path)
        try:
            np.save(cache_file, features)
        except Exception:
            pass
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling class imbalance."""
        labels = [l for _, l in self.samples]
        n_genuine = sum(labels)
        n_fake = len(labels) - n_genuine
        total = len(labels)
        
        # Inverse frequency weighting
        weight_fake = total / (2 * n_fake) if n_fake > 0 else 1.0
        weight_genuine = total / (2 * n_genuine) if n_genuine > 0 else 1.0
        
        return torch.tensor([weight_fake, weight_genuine], dtype=torch.float32)
    
    def get_pos_weight(self) -> torch.Tensor:
        """Get positive class weight for BCEWithLogitsLoss."""
        labels = [l for _, l in self.samples]
        n_genuine = sum(labels)
        n_fake = len(labels) - n_genuine
        
        if n_genuine == 0:
            return torch.tensor([1.0])
        
        pos_weight = n_fake / n_genuine
        return torch.tensor([pos_weight], dtype=torch.float32)


def create_dataloaders(data_dir: str, batch_size: int = 16,
                       num_workers: int = 4, cache_dir: str = None,
                       max_samples: int = None) -> dict:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Root data directory (e.g., 'data/for-norm/')
        batch_size: Batch size (16 recommended for RTX 3050 4GB)
        num_workers: Number of data loading workers
        cache_dir: Directory for feature caching
        max_samples: Limit samples per split (for debugging)
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders and 'train_dataset'
    """
    data_path = Path(data_dir)
    
    # Create datasets
    print("\nLoading datasets:")
    train_dataset = DeepfakeAudioDataset(
        root_dir=str(data_path / "training"),
        augment=True,
        cache_dir=os.path.join(cache_dir, "train") if cache_dir else None,
        max_samples=max_samples
    )
    
    val_dataset = DeepfakeAudioDataset(
        root_dir=str(data_path / "validation"),
        augment=False,
        cache_dir=os.path.join(cache_dir, "val") if cache_dir else None,
        max_samples=max_samples // 4 if max_samples else None
    )
    
    test_dataset = DeepfakeAudioDataset(
        root_dir=str(data_path / "testing"),
        augment=False,
        cache_dir=os.path.join(cache_dir, "test") if cache_dir else None,
        max_samples=max_samples // 4 if max_samples else None
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
    }
