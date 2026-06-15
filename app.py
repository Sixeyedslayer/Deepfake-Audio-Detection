"""
app.py — Streamlit web application for deepfake audio detection.

Features:
  - Audio file upload (WAV, MP3, FLAC, OGG)
  - Audio playback
  - Waveform visualization
  - Mel-spectrogram visualization
  - MFCC heatmap
  - Classification result with confidence gauge
  - Model information sidebar

Usage:
    streamlit run app.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import librosa
import librosa.display

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DeepfakeDetector, build_model
from src.preprocessing import (
    load_audio, preprocess_audio, extract_features,
    SAMPLE_RATE, N_MFCC, N_MELS, N_FFT, HOP_LENGTH
)


# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }
    
    /* Result cards */
    .result-genuine {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }
    
    .result-fake {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }
    
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    
    .result-confidence {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
    }
    
    .confidence-genuine { color: #34d399; }
    .confidence-fake { color: #f87171; }
    
    .result-sub {
        color: #d1d5db;
        font-size: 0.95rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        padding: 0.5rem 0;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Info cards */
    .info-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg { padding-top: 2rem; }
    
    /* Upload area */
    .uploadedFile { border: 2px dashed #6366f1 !important; }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (Cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model (cached across sessions)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DeepfakeDetector(
        num_channels=4,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
        pretrained_resnet=False
    )
    
    model_path = Path(__file__).parent / "models" / "best_model.pth"
    
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_info = {
            "loaded": True,
            "epoch": checkpoint.get("epoch", "N/A"),
            "val_eer": checkpoint.get("val_eer", "N/A"),
            "val_accuracy": checkpoint.get("val_accuracy", "N/A"),
        }
    else:
        model_info = {"loaded": False}
    
    model = model.to(device)
    model.eval()
    
    return model, device, model_info


# ─── Visualization Functions ─────────────────────────────────────────────────
def plot_waveform(y, sr):
    """Generate waveform plot."""
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    
    # Create gradient effect
    ax.fill_between(time_axis, y, alpha=0.3, color="#6366f1")
    ax.plot(time_axis, y, color="#818cf8", linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=11)
    ax.set_title("Waveform", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#64748b")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, len(y) / sr)
    
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(y, sr):
    """Generate mel-spectrogram plot."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(
        log_mel, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax,
        cmap="magma"
    )
    
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    cbar.outline.set_edgecolor("#334155")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
    
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=11)
    ax.set_title("Mel Spectrogram", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#64748b")
    
    plt.tight_layout()
    return fig


def plot_mfcc(y, sr):
    """Generate MFCC heatmap."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    img = librosa.display.specshow(
        mfcc, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", ax=ax,
        cmap="coolwarm"
    )
    
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    cbar.outline.set_edgecolor("#334155")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
    
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("MFCC Coefficient", color="#94a3b8", fontsize=11)
    ax.set_title("MFCC Features", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#64748b")
    
    plt.tight_layout()
    return fig


# ─── Prediction Function ────────────────────────────────────────────────────
@torch.no_grad()
def predict_audio(model, audio_path, device):
    """Run inference on an audio file."""
    start = time.time()
    
    # Load and preprocess
    y = load_audio(audio_path)
    y_processed = preprocess_audio(y)
    
    # Extract features
    features = extract_features(y_processed)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    # Inference
    logits = model(features_tensor)
    score = torch.sigmoid(logits).item()
    
    # Result
    is_genuine = score >= 0.5
    confidence = score if is_genuine else (1 - score)
    elapsed = time.time() - start
    
    return {
        "prediction": "Genuine (Human)" if is_genuine else "Deepfake (AI-Generated)",
        "is_genuine": is_genuine,
        "confidence": confidence,
        "score": score,
        "time_ms": elapsed * 1000,
        "audio_raw": y,
        "audio_processed": y_processed,
    }


# ─── Main App ───────────────────────────────────────────────────────────────
def main():
    # Load model
    model, device, model_info = load_model()
    
    # ─── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Model Information")
        
        if model_info.get("loaded"):
            st.success("Model loaded successfully!")
            st.markdown(f"**Device:** `{device}`")
            st.markdown(f"**Training Epoch:** `{model_info['epoch']}`")
            if isinstance(model_info.get("val_eer"), float):
                st.markdown(f"**Validation EER:** `{model_info['val_eer']:.4f}`")
            if isinstance(model_info.get("val_accuracy"), float):
                st.markdown(f"**Validation Accuracy:** `{model_info['val_accuracy']:.2%}`")
        else:
            st.warning("⚠️ No trained model found!\n\nPlease train the model first:\n```bash\npython -m src.train\n```")
        
        st.markdown("---")
        st.markdown("## 📖 About")
        st.markdown("""
        This app uses a **hybrid ResNet-BiLSTM** architecture to detect 
        AI-generated (deepfake) audio.
        
        **Features analyzed:**
        - Mel-Spectrogram patterns
        - MFCC coefficients + deltas
        - Temporal consistency
        
        **Architecture:**
        - ResNet-18 spatial features
        - BiLSTM temporal modeling
        - Attention-based pooling
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
            Built with PyTorch & Streamlit<br>
            Deepfake Audio Detection Project
        </div>
        """, unsafe_allow_html=True)
    
    # ─── Main Content ────────────────────────────────────────────────
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Deepfake Audio Detector</h1>
        <p>Upload an audio file to detect if it's genuine human speech or AI-generated</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="section-header">📁 Upload Audio</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Audio player
            st.markdown('<div class="section-header">🔊 Audio Playback</div>',
                        unsafe_allow_html=True)
            st.audio(uploaded_file, format=f"audio/{suffix.lstrip('.')}")
            
            # Run prediction
            if not model_info.get("loaded"):
                st.error("❌ Cannot run prediction: No trained model found. "
                         "Please train the model first.")
                return
            
            with st.spinner("🔍 Analyzing audio..."):
                result = predict_audio(model, tmp_path, device)
            
            # ─── Result Display ──────────────────────────────────────
            st.markdown('<div class="section-header">🎯 Detection Result</div>',
                        unsafe_allow_html=True)
            
            if result["is_genuine"]:
                st.markdown(f"""
                <div class="result-genuine">
                    <p class="result-label">✅ {result['prediction']}</p>
                    <p class="result-confidence confidence-genuine">{result['confidence']:.1%}</p>
                    <p class="result-sub">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fake">
                    <p class="result-label">🔴 {result['prediction']}</p>
                    <p class="result-confidence confidence-fake">{result['confidence']:.1%}</p>
                    <p class="result-sub">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics row
            st.markdown("")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Score", f"{result['score']:.4f}")
            with col2:
                st.metric("Processing Time", f"{result['time_ms']:.0f} ms")
            with col3:
                duration = len(result["audio_raw"]) / SAMPLE_RATE
                st.metric("Audio Duration", f"{duration:.2f} s")
            
            # ─── Visualizations ──────────────────────────────────────
            st.markdown("")
            st.markdown('<div class="section-header">📊 Audio Analysis</div>',
                        unsafe_allow_html=True)
            
            # Waveform
            fig_wave = plot_waveform(result["audio_processed"], SAMPLE_RATE)
            st.pyplot(fig_wave)
            plt.close(fig_wave)
            
            # Spectrogram and MFCC in tabs
            tab1, tab2 = st.tabs(["🌈 Mel Spectrogram", "📈 MFCC Features"])
            
            with tab1:
                fig_mel = plot_mel_spectrogram(result["audio_processed"], SAMPLE_RATE)
                st.pyplot(fig_mel)
                plt.close(fig_mel)
            
            with tab2:
                fig_mfcc = plot_mfcc(result["audio_processed"], SAMPLE_RATE)
                st.pyplot(fig_mfcc)
                plt.close(fig_mfcc)
        
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #64748b;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎤</div>
            <h3 style="color: #94a3b8;">Upload an audio file to get started</h3>
            <p>The detector will analyze the audio and determine if it's<br>
            genuine human speech or AI-generated deepfake audio.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
