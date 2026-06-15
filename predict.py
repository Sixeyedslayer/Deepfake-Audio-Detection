"""
predict.py — Command-line inference script for deepfake audio detection.

Test the trained model on new audio samples.

Usage:
    # Single file
    python predict.py --audio path/to/audio.wav

    # Directory of files
    python predict.py --audio path/to/directory/

    # Custom model path
    python predict.py --audio audio.wav --model models/best_model.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.model import DeepfakeDetector, build_model
from src.preprocessing import extract_features_for_inference
from src.utils import get_device


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"  Loading model from: {model_path}")
    
    model = build_model(device=device, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    epoch = checkpoint.get("epoch", "?")
    val_eer = checkpoint.get("val_eer", "?")
    print(f"  Checkpoint: epoch={epoch}, val_eer={val_eer}")
    
    return model


@torch.no_grad()
def predict_single(model, audio_path: str, device: torch.device) -> dict:
    """
    Predict whether a single audio file is genuine or deepfake.
    
    Returns:
        Dictionary with prediction results.
    """
    start_time = time.time()
    
    # Extract features
    features = extract_features_for_inference(audio_path)
    features = features.to(device)
    
    # Inference
    logits = model(features)
    score = torch.sigmoid(logits).item()
    
    # Classification
    is_genuine = score >= 0.5
    confidence = score if is_genuine else (1 - score)
    
    elapsed = time.time() - start_time
    
    return {
        "file": os.path.basename(audio_path),
        "path": audio_path,
        "prediction": "Genuine (Human)" if is_genuine else "Deepfake (AI-Generated)",
        "label": "genuine" if is_genuine else "fake",
        "confidence": float(confidence),
        "score": float(score),
        "processing_time_ms": float(elapsed * 1000),
    }


def predict_directory(model, dir_path: str, device: torch.device) -> list:
    """Predict on all audio files in a directory."""
    audio_files = sorted([
        str(f) for f in Path(dir_path).rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    
    if not audio_files:
        print(f"  No audio files found in: {dir_path}")
        return []
    
    print(f"  Found {len(audio_files)} audio files")
    print()
    
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        try:
            result = predict_single(model, audio_path, device)
            results.append(result)
            
            # Print result
            icon = "✅" if result["label"] == "genuine" else "🔴"
            print(f"  [{i:03d}/{len(audio_files)}] {icon} {result['file']}")
            print(f"           → {result['prediction']} "
                  f"(confidence: {result['confidence']:.1%}, "
                  f"time: {result['processing_time_ms']:.0f}ms)")
            
        except Exception as e:
            print(f"  [{i:03d}/{len(audio_files)}] ❌ ERROR: {audio_path}")
            print(f"           → {str(e)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection — Inference")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file or directory")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  DEEPFAKE AUDIO DETECTION — INFERENCE")
    print("=" * 60)
    print()
    
    # Setup
    device = get_device()
    model = load_model(args.model, device)
    
    print()
    
    # Check if input is file or directory
    target = Path(args.audio)
    
    if target.is_file():
        # Single file prediction
        result = predict_single(model, str(target), device)
        
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  File: {result['file']:<33s}│")
        print("  ├─────────────────────────────────────────┤")
        
        if result["label"] == "genuine":
            print(f"  │  ✅ GENUINE (Human)                      │")
        else:
            print(f"  │  🔴 DEEPFAKE (AI-Generated)              │")
        
        print(f"  │  Confidence: {result['confidence']:.1%}                       │")
        print(f"  │  Raw Score:  {result['score']:.4f}                       │")
        print(f"  │  Time:       {result['processing_time_ms']:.0f}ms                          │")
        print("  └─────────────────────────────────────────┘")
    
    elif target.is_dir():
        # Directory prediction
        results = predict_directory(model, str(target), device)
        
        if results:
            # Summary
            n_genuine = sum(1 for r in results if r["label"] == "genuine")
            n_fake = sum(1 for r in results if r["label"] == "fake")
            avg_conf = np.mean([r["confidence"] for r in results])
            avg_time = np.mean([r["processing_time_ms"] for r in results])
            
            print(f"\n  Summary: {n_genuine} genuine, {n_fake} fake "
                  f"(avg confidence: {avg_conf:.1%}, avg time: {avg_time:.0f}ms)")
    
    else:
        print(f"  ERROR: Path not found: {args.audio}")
        sys.exit(1)
    
    print()


if __name__ == "__main__":
    main()
