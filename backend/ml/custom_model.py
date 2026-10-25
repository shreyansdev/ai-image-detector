#!/usr/bin/env python3
"""
custom_model.py - ForensicNet: A Multi-Stream Neural Network for AI Image Detection.

Forensic Rationale:
1. Bayar Constrained Convolution Stream (Noise/Residual):
   Forces the network to see pixel prediction-error patterns rather than semantic content,
   since generation artifacts (e.g., GAN noise patterns, subtle sensor noise inconsistencies,
   and resampling traces) often live in these high-frequency residuals.
   
2. Frequency Domain Stream (2D FFT Spectrum):
   GAN and diffusion upsampling operations (such as transposed convolutions or bilinear/bicubic
   interpolation) leave detectable periodic artifacts and structural grid anomalies in the
   frequency spectrum that are invisible or diffuse in the spatial domain.
   
3. Spatial / Semantic Stream (Raw RGB):
   Pure forensic signals alone can miss cases where semantic content itself is a stronger cue
   (e.g., anatomically implausible details, distorted hands, impossible lighting, or unnatural textures).
   Adding a spatial semantic stream gives the model holistic visual context to complement the forensic streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BayarConv2d(nn.Module):
    """
    Bayar Constrained Convolutional Layer for image forensics.
    
    Reference:
        Bayar & Stamm, "A Deep Learning Approach to Universal Image Manipulation Detection", IH&MMSec 2016.
        
    Constraint Rationale:
        For each 5x5 filter kernel w in the layer:
        - The center weight w(2,2) is fixed to -1.
        - All other 24 weights in that filter are renormalized to sum to +1.
        - Consequently, the total filter weights sum to zero: sum(w) = +1 - 1 = 0.
        This forces the filter to suppress low-frequency scene content and extract pixel
        prediction errors (residuals) between a target pixel and its neighborhood.
        
    Enforcement Strategy:
        Constructs the constrained weight tensor non-in-place during forward pass,
        avoiding in-place parameter mutations that interfere with GradScaler/optimizer tracking.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 4, padding: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        
        # 5x5 convolution weight parameter without bias
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=1, padding=padding, bias=False
        )
        
        # Initialize weights with Xavier uniform distribution
        nn.init.xavier_uniform_(self.conv.weight)

    def constrain_weights(self) -> torch.Tensor:
        """
        Enforces center weight = -1 and non-center weight sum = +1 across all 5x5 filters non-in-place.
        Returns the constrained weight tensor without modifying `self.conv.weight.data` in-place.
        """
        w = self.conv.weight
        mask = torch.ones_like(w)
        mask[:, :, 2, 2] = 0.0
        w_zero_center = w * mask
        sum_w = w_zero_center.sum(dim=(-2, -1), keepdim=True)
        sum_w = torch.where(sum_w == 0, torch.tensor(1.0, device=w.device, dtype=w.dtype), sum_w)
        w_norm = w_zero_center / sum_w
        center = torch.zeros_like(w)
        center[:, :, 2, 2] = -1.0
        return w_norm + center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constrained_weight = self.constrain_weights()
        return F.conv2d(x, constrained_weight, padding=self.padding, stride=1)


def compute_fft_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D FFT log-magnitude spectrum of an input image batch in FP32.
    
    Transform Sequence:
    1. 2D Discrete Fast Fourier Transform: torch.fft.fft2(x, dim=(-2, -1))
    2. Magnitude spectrum computation: |FFT(x)|
    3. Logarithmic compression for numerical stability: log(1 + magnitude)
    4. Frequency shift: torch.fft.fftshift moves the zero-frequency DC component to center
    5. Min-Max normalization to [0, 1] range per sample with epsilon=1e-6 and clamping
    
    Args:
        x: Image tensor of shape (B, C, H, W).
        
    Returns:
        Normalized log-magnitude frequency spectrum tensor of shape (B, C, H, W).
    """
    orig_dtype = x.dtype
    device_type = 'cuda' if x.is_cuda else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        x_f32 = x.float()
        fft_result = torch.fft.fft2(x_f32, dim=(-2, -1))
        magnitude = torch.abs(fft_result)
        log_magnitude = torch.log(1.0 + magnitude)
        shifted_spectrum = torch.fft.fftshift(log_magnitude, dim=(-2, -1))
        
        # Min-max normalization per sample across channels and spatial dimensions
        amin = shifted_spectrum.amin(dim=(-3, -2, -1), keepdim=True)
        amax = shifted_spectrum.amax(dim=(-3, -2, -1), keepdim=True)
        normalized_spectrum = (shifted_spectrum - amin) / (amax - amin + 1e-6)
        normalized_spectrum = torch.clamp(normalized_spectrum, 0.0, 1.0)
        
    return normalized_spectrum.to(orig_dtype)


class ConvBlock(nn.Module):
    """
    Standard Conv-BatchNorm-ReLU-MaxPool block used across spatial and frequency streams.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ForensicNet(nn.Module):
    """
    Dual/Tri-Stream Forensic Convolutional Neural Network for AI Image Detection.
    
    Architecture Overview:
    - STREAM A (Noise/Residual): BayarConv2d -> 5 Conv Blocks (16->32->64->128->256) -> GAP -> Vector (256)
    - STREAM B (Frequency): FFT Preprocessing -> 4 Conv Blocks (16->32->64->128) -> GAP -> Vector (128)
    - STREAM C (Spatial/Semantic): Raw RGB -> 5 Conv Blocks (16->32->64->128->256) -> GAP -> Vector (256)
    - FUSION: Concatenation (640-dim) -> Linear(640, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 2)
    """
    def __init__(self, num_classes: int = 2, bayar_channels: int = 4):
        super().__init__()
        
        # STREAM A: Bayar Constrained Noise/Residual Stream
        self.bayar_conv = BayarConv2d(in_channels=3, out_channels=bayar_channels, padding=2)
        self.stream_a = nn.Sequential(
            ConvBlock(bayar_channels, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # STREAM B: Frequency Domain Stream
        self.stream_b = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # STREAM C: Spatial / Semantic Stream (Raw RGB)
        self.stream_c = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # FUSION CLASSIFIER
        fusion_dim = 256 + 128 + 256  # 640 total features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Print parameter summary
        param_count = self.count_parameters()
        print(f"[ForensicNet] Initialized custom architecture. Total trainable parameters: {param_count:,}")

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def constrain_bayar_weights(self) -> torch.Tensor:
        """
        Public helper to compute and return Bayar constrained weights.
        Note: Weight constraint is automatically applied non-in-place during forward pass.
        """
        return self.bayar_conv.constrain_weights()

    def extract_stream_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts flattened feature vectors from Stream A, Stream B, and Stream C."""
        # Stream A: Bayar residual filtering
        res_a = self.bayar_conv(x)
        feat_a = torch.flatten(self.stream_a(res_a), start_dim=1)
        
        # Stream B: Frequency domain log-magnitude spectrum
        freq_b = compute_fft_spectrum(x)
        feat_b = torch.flatten(self.stream_b(freq_b), start_dim=1)
        
        # Stream C: Spatial RGB content
        feat_c = torch.flatten(self.stream_c(x), start_dim=1)
        
        return feat_a, feat_b, feat_c

    def get_stream_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Debug method returning intermediate feature vectors from each stream.
        
        Returns:
            Dictionary with keys 'stream_a', 'stream_b', 'stream_c'.
        """
        feat_a, feat_b, feat_c = self.extract_stream_features(x)
        return {
            "stream_a": feat_a,
            "stream_b": feat_b,
            "stream_c": feat_c
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_a, feat_b, feat_c = self.extract_stream_features(x)
        fused = torch.cat([feat_a, feat_b, feat_c], dim=1)
        output = self.classifier(fused)
        return output


if __name__ == "__main__":
    print("=== Running ForensicNet Self-Test ===")
    
    # 1. Instantiate model
    model = ForensicNet(num_classes=2)
    
    # 2. Verify Bayar constraint math
    bayar_w = model.constrain_bayar_weights()
    # Check center weight == -1
    center_val = bayar_w[0, 0, 2, 2].item()
    # Check sum of total filter weights == 0 (tolerance 1e-5)
    filter_sum = bayar_w[0, 0].sum().item()
    print(f"Bayar filter check: center weight = {center_val:.4f}, total sum = {filter_sum:.6f}")
    assert abs(center_val - (-1.0)) < 1e-5, f"Center weight constraint failed: {center_val}"
    assert abs(filter_sum) < 1e-5, f"Filter zero sum constraint failed: {filter_sum}"
    
    # 3. Test dummy forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input batch shape:  {dummy_input.shape}")
    print(f"Output logits shape: {output.shape}")
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"
    
    # 4. Test debug get_stream_outputs method
    stream_outputs = model.get_stream_outputs(dummy_input)
    print("Stream Feature Vectors:")
    for k, v in stream_outputs.items():
        print(f"  - {k}: shape {v.shape}")
    assert stream_outputs["stream_a"].shape == (4, 256)
    assert stream_outputs["stream_b"].shape == (4, 128)
    assert stream_outputs["stream_c"].shape == (4, 256)
    
    print("=== All ForensicNet Self-Tests Passed Successfully ===")
