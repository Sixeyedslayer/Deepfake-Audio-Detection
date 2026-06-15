"""
model.py — Hybrid ResNet-BiLSTM architecture for deepfake audio detection.

Architecture:
  Input (B, 4, 128, T) 
    → ResNet-18 backbone (modified first conv for 4 channels)
    → Feature maps (B, 512, H, W) 
    → Reshape to sequence (B, W, 512*H)
    → BiLSTM (2 layers, hidden=256) 
    → Attention pooling 
    → FC layers → Sigmoid
    → P(genuine)

Total parameters: ~14M
Optimized for RTX 3050 (4-8 GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over time steps.
    Learns to weight time steps by their importance, rather than
    naive mean/max pooling.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            Weighted sum: (batch, features)
        """
        # Compute attention weights
        attn_weights = self.attention(x)          # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T, 1)
        
        # Weighted sum
        out = torch.sum(x * attn_weights, dim=1)  # (B, features)
        return out


class DeepfakeDetector(nn.Module):
    """
    Hybrid ResNet-18 + BiLSTM model for audio deepfake detection.
    
    The ResNet extracts spatial features from the multi-channel 
    spectrogram input, the BiLSTM captures temporal dependencies,
    and attention pooling focuses on the most discriminative segments.
    
    Args:
        num_channels: Number of input channels (4: mel + mfcc + delta + delta2)
        lstm_hidden: Hidden size for each direction of BiLSTM
        lstm_layers: Number of BiLSTM layers
        dropout: Dropout probability
        pretrained_resnet: Whether to use ImageNet-pretrained weights
    """
    
    def __init__(
        self,
        num_channels: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        pretrained_resnet: bool = True
    ):
        super().__init__()
        
        # ─── ResNet-18 Backbone ──────────────────────────────────────
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained_resnet else None)
        
        # Modify first conv layer: 3 channels → num_channels
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new conv weights by averaging pretrained weights
        if pretrained_resnet:
            with torch.no_grad():
                # Average the 3-channel weights and repeat for num_channels
                mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
                self.conv1.weight.copy_(mean_weight.repeat(1, num_channels, 1, 1))
        
        # Use rest of ResNet layers (excluding fc and avgpool)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels
        
        # Adaptive pool to reduce spatial dimensions to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # Both height and width fixed for consistent LSTM input
        
        # ─── BiLSTM Temporal Modeling ────────────────────────────────
        self.lstm_input_size = 512 * 4  # 512 channels × 4 height
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # ─── Attention Pooling ───────────────────────────────────────
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.attention = AttentionPooling(lstm_output_size)
        
        # ─── Classification Head ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classification head with Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, freq_bins, time_steps)
               Typically (B, 4, 128, T)
        
        Returns:
            logits: (batch, 1) — raw logits (apply sigmoid for probability)
        """
        # ─── ResNet Feature Extraction ───────────────────────────────
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x shape: (B, 512, H', W')
        
        # Adaptive pool: fix height, preserve width (time)
        x = self.adaptive_pool(x)
        # x shape: (B, 512, 4, W')
        
        # ─── Reshape for LSTM ────────────────────────────────────────
        batch, channels, height, width = x.shape
        # Treat width as time steps, flatten channel × height as features
        x = x.permute(0, 3, 1, 2)            # (B, W', 512, 4)
        x = x.reshape(batch, width, -1)       # (B, W', 2048)
        
        # ─── BiLSTM ─────────────────────────────────────────────────
        x, _ = self.lstm(x)
        # x shape: (B, W', lstm_hidden*2)
        
        # ─── Attention Pooling ───────────────────────────────────────
        x = self.attention(x)
        # x shape: (B, lstm_hidden*2)
        
        # ─── Classification ─────────────────────────────────────────
        logits = self.classifier(x)
        # logits shape: (B, 1)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability of being genuine.
        
        Args:
            x: Input tensor (B, C, F, T)
        
        Returns:
            Probability tensor (B, 1) in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    @staticmethod
    def count_parameters(model: nn.Module) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_mb": total * 4 / (1024 ** 2),  # Assuming float32
        }


def build_model(device: torch.device = None, pretrained: bool = True) -> DeepfakeDetector:
    """
    Build and return the DeepfakeDetector model.
    
    Args:
        device: Target device (cuda/cpu)
        pretrained: Use ImageNet-pretrained ResNet weights
    
    Returns:
        Model moved to device
    """
    model = DeepfakeDetector(
        num_channels=4,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
        pretrained_resnet=pretrained
    )
    
    if device:
        model = model.to(device)
    
    # Print model summary
    params = DeepfakeDetector.count_parameters(model)
    print(f"\n  Model: DeepfakeDetector (ResNet-18 + BiLSTM + Attention)")
    print(f"  Total parameters: {params['total']:,} ({params['total_mb']:.1f} MB)")
    print(f"  Trainable: {params['trainable']:,}")
    
    return model
