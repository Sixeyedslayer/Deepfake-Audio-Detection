"""Quick verification script to test all imports and model build."""
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import librosa
print(f"librosa: {librosa.__version__}")

import sklearn
print(f"scikit-learn: {sklearn.__version__}")

import streamlit
print(f"streamlit: {streamlit.__version__}")

import scipy
print(f"scipy: {scipy.__version__}")

# Test model build
print("\nBuilding model...")
from src.model import build_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(device=device, pretrained=True)

# Test forward pass with dummy input
print("\nTesting forward pass...")
dummy = torch.randn(2, 4, 128, 126).to(device)
with torch.no_grad():
    output = model(dummy)
print(f"Input shape:  {dummy.shape}")
print(f"Output shape: {output.shape}")
print(f"Output values: {torch.sigmoid(output).flatten().tolist()}")

print("\n=== ALL CHECKS PASSED ===")
