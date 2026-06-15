# 🎭 Deepfake Audio Detection

A deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using a hybrid ResNet-BiLSTM architecture with attention pooling.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📦 Deliverables Checklist

- [ ] **GITHUB REPOSITORY**: Codebase ready (Push pending)
- [ ] **STREAMLIT WEB APP (HOSTED)**: Ready for Hugging Face Spaces deployment
- [x] **ipynb notebook with full running code**: `Deepfake_Audio_Detection.ipynb`
- [x] **Trained model**: `models/best_model.pth`
- [x] **Python script for testing new samples**: `predict.py`
- [x] **Performance report**: Found in `reports/` and printed to console.
- [x] **Description of preprocessing, feature extraction, and model architecture**: See [Architecture](#-architecture)
- [x] **Clear and detailed README.md**: You're reading it!
- [x] **Receives an audio file as input**: Fully supported via CLI and Web App.
- [x] **Returns Genuine/Deepfake + confidence score**: Fully supported.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Streamlit Web App](#streamlit-web-app)
- [Project Structure](#project-structure)
- [Results](#results)
- [Deployment](#deployment)

---

## 🔍 Overview

Advances in generative AI have enabled the creation of highly realistic synthetic speech (deepfake audio), which can be misused for impersonation, fraud, and misinformation. This project develops a robust detector to identify AI-generated audio.

### Key Features
- **Hybrid Architecture**: ResNet-18 (spatial) + BiLSTM (temporal) + Attention Pooling
- **Multi-Feature Fusion**: Mel-spectrogram + MFCC + Delta + Delta-Delta
- **Data Augmentation**: SpecAugment, noise injection, time/pitch shifting
- **Mixed Precision Training**: FP16 for efficient GPU utilization
- **Interactive Web App**: Streamlit-based with real-time audio analysis

---

## 🏗️ Architecture

```
Input Audio (.wav)
    │
    ▼
┌─────────────────────┐
│   Preprocessing     │  ← 16kHz, trim silence, pad/truncate to 4s
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Extraction │  ← Mel-Spectrogram (128 bins)
│                     │  ← MFCC (40 coeffs) + Δ + ΔΔ
│                     │  → 4-channel tensor (4, 128, T)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ResNet-18         │  ← Modified for 4 input channels
│   (Spatial)         │  ← ImageNet pretrained backbone
│                     │  → Feature maps (512, H, W)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   BiLSTM            │  ← 2 layers, hidden=256
│   (Temporal)        │  ← Captures prosodic patterns
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Attention Pooling  │  ← Learnable time-step weighting
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Classifier (FC)    │  ← LayerNorm + Dropout + FC
│                     │  → Sigmoid → P(Genuine)
└─────────────────────┘
```

**Total Parameters**: ~14M | **Model Size**: ~56 MB

---

## 📊 Dataset

**Primary**: [The Fake-or-Real (FoR) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

- **Version Used**: `for-norm` (normalized, balanced)
- **Format**: WAV files at 16 kHz
- **Classes**: `real/` (genuine human speech) and `fake/` (AI-generated speech)
- **Splits**: Training, Validation, Testing

### Setup

1. Install the Kaggle API: `pip install kaggle`
2. Place your Kaggle API token at `~/.kaggle/kaggle.json`
3. Run the setup script:

```bash
python setup_dataset.py
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)

### Steps

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deepfake-audio-detection.git
cd deepfake-audio-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download dataset
python setup_dataset.py
```

---

## 🏋️ Training

```bash
# Default training (30 epochs, batch_size=16, lr=1e-4)
python -m src.train

# Custom training
python -m src.train --epochs 50 --batch_size 32 --lr 5e-5

# Quick test run (limited samples)
python -m src.train --max_samples 200 --epochs 5
```

### Training Configuration

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 16 | Batch size (16 for RTX 3050) |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--patience` | 7 | Early stopping patience |
| `--num_workers` | 2 | DataLoader workers |

---

## 📈 Evaluation

```bash
# Evaluate on test set
python -m src.evaluate --model models/best_model.pth

# Custom data directory
python -m src.evaluate --model models/best_model.pth --data_dir data/for-norm
```

### Metrics Computed
- ✅ Overall Accuracy
- ✅ Equal Error Rate (EER)
- ✅ F1 Score
- ✅ Precision & Recall
- ✅ Per-class Accuracy
- ✅ Confusion Matrix
- ✅ ROC Curve + AUC
- ✅ DET Curve
- ✅ Score Distribution

Reports are saved to `reports/`.

---

## 🔮 Inference

```bash
# Single file
python predict.py --audio path/to/audio.wav

# Batch (entire directory)
python predict.py --audio path/to/audio_folder/

# Custom model
python predict.py --audio audio.wav --model models/best_model.pth
```

---

## 🌐 Streamlit Web App

### Run Locally

```bash
streamlit run app.py
```

### Features
- 📁 Audio file upload (WAV, MP3, FLAC, OGG)
- 🔊 In-browser audio playback
- 📊 Waveform visualization
- 🌈 Mel-spectrogram display
- 📈 MFCC feature heatmap
- 🎯 Classification result with confidence score
- ⚡ Real-time processing

### Deploy to Hugging Face Spaces

The app is configured for deployment on [Hugging Face Spaces](https://huggingface.co/spaces).

---

## 📁 Project Structure

```
deepfake-audio-detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── setup_dataset.py             # Dataset download & verification
├── predict.py                   # CLI inference script
├── app.py                       # Streamlit web application
├── deepfake_detection.ipynb     # Complete Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Audio loading & feature extraction
│   ├── dataset.py               # PyTorch Dataset & DataLoader
│   ├── model.py                 # ResNet-BiLSTM architecture
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation & metrics
│   └── utils.py                 # Helper functions (EER, seeding, etc.)
│
├── models/                      # Saved model checkpoints
│   ├── best_model.pth
│   └── training_history.json
│
├── data/                        # Dataset (not in git)
│   └── for-norm/
│       ├── training/{real,fake}/
│       ├── validation/{real,fake}/
│       └── testing/{real,fake}/
│
└── reports/                     # Evaluation outputs
    ├── metrics.json
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── det_curve.png
    └── score_distribution.png
```

---

## 📊 Results

### Performance on FoR Test Set

| Metric | Value | Threshold |
|---|---|---|
| Overall Accuracy | **91.52%** | ≥ 80% |
| Equal Error Rate (EER) | **2.57%** | ≤ 12% |
| F1 Score | **92.01%** | ≥ 80% |
| Per-class Acc (Genuine) | **99.91%** | ≥ 75% |
| Per-class Acc (Deepfake) | **83.50%** | ≥ 75% |
| ROC AUC | **0.9865** | — |

*Results based on evaluation of 4,634 completely unseen test samples after training for 24 epochs on the FoR dataset.*

---

## 🚀 Deployment

### Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Push this repository to the Space
4. The app will be automatically deployed

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) by Mohammed Abdeldayem
- [ASVspoof 2019](https://www.asvspoof.org/index2019.html) benchmark
- PyTorch, librosa, Streamlit, scikit-learn
