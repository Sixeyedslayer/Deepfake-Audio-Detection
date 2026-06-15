"""
preprocessing.py — Audio loading, preprocessing, and feature extraction.

Pipeline:
  1. Load WAV at 16 kHz mono
  2. Trim silence
  3. Pad/truncate to fixed duration (4 seconds = 64,000 samples)
  4. Normalize amplitude
  5. Extract features: MFCC (40 coeffs + Δ + ΔΔ) and Mel-Spectrogram (128 bins)
"""

import numpy as np
import librosa
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Constants ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION = 4  # seconds
MAX_SAMPLES = SAMPLE_RATE * DURATION  # 64,000

# Feature extraction parameters
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and return as a 1D numpy array.
    
    Args:
        file_path: Path to the audio file (WAV, MP3, FLAC, OGG)
        sr: Target sample rate
    
    Returns:
        Audio signal as numpy array
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")
    
    return y


def preprocess_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Preprocess audio: trim silence, pad/truncate, normalize.
    
    Args:
        y: Raw audio signal
        sr: Sample rate
    
    Returns:
        Preprocessed audio signal of fixed length
    """
    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # If trimming removed everything, use original
    if len(y_trimmed) < sr // 4:  # less than 0.25s
        y_trimmed = y
    
    # Pad or truncate to fixed length
    if len(y_trimmed) < MAX_SAMPLES:
        # Pad with zeros (silence) at the end
        y_trimmed = np.pad(y_trimmed, (0, MAX_SAMPLES - len(y_trimmed)), mode="constant")
    else:
        # Truncate from the center for more representative content
        start = (len(y_trimmed) - MAX_SAMPLES) // 2
        y_trimmed = y_trimmed[start : start + MAX_SAMPLES]
    
    # Normalize amplitude to [-1, 1]
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_trimmed = y_trimmed / max_val
    
    return y_trimmed


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCC features with delta and delta-delta.
    
    Args:
        y: Preprocessed audio signal
        sr: Sample rate
    
    Returns:
        MFCC features with shape (3, n_mfcc, time_steps)
        Channel 0: MFCC, Channel 1: Delta, Channel 2: Delta-Delta
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    
    # Compute deltas
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack: (3, n_mfcc, time_steps)
    features = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    return features


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract log-Mel spectrogram.
    
    Args:
        y: Preprocessed audio signal
        sr: Sample rate
    
    Returns:
        Log-Mel spectrogram with shape (1, n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    
    # Add channel dimension: (1, n_mels, time_steps)
    return log_mel[np.newaxis, ...]


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract combined features: Mel-spectrogram + MFCC (resized).
    
    The MFCC channels (40 coefficients) are interpolated to match
    the Mel-spectrogram dimensions (128 bins), producing a 
    4-channel input tensor.
    
    Args:
        y: Preprocessed audio signal
        sr: Sample rate
    
    Returns:
        Combined features with shape (4, 128, time_steps)
    """
    # Extract individual features
    mel = extract_mel_spectrogram(y, sr)  # (1, 128, T)
    mfcc = extract_mfcc(y, sr)            # (3, 40, T)
    
    # Resize MFCC from 40 bins to 128 bins to match mel dimensions
    # Using simple interpolation along the frequency axis
    from scipy.ndimage import zoom
    
    scale_factor = N_MELS / N_MFCC  # 128/40 = 3.2
    mfcc_resized = zoom(mfcc, (1, scale_factor, 1), order=1)  # (3, 128, T)
    
    # Ensure exact shape match
    time_steps = min(mel.shape[2], mfcc_resized.shape[2])
    mel = mel[:, :, :time_steps]
    mfcc_resized = mfcc_resized[:, :, :time_steps]
    
    # Concatenate: (4, 128, T)
    combined = np.concatenate([mel, mfcc_resized], axis=0)
    
    return combined.astype(np.float32)


def extract_features_for_inference(file_path: str) -> torch.Tensor:
    """
    Complete pipeline: load audio → preprocess → extract features → tensor.
    Used by predict.py and app.py for single-file inference.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Feature tensor with shape (1, 4, 128, T) ready for model input
    """
    y = load_audio(file_path)
    y = preprocess_audio(y)
    features = extract_features(y)
    tensor = torch.from_numpy(features).unsqueeze(0)  # Add batch dim
    return tensor
