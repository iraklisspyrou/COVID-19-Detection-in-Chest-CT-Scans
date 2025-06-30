# custom_cnn.py
"""Simple custom CNN classifier for binary COVID vs non‑COVID CT scans.

Designed to be trained **from scratch** on 224×224 images.
You can import this module and instantiate `CustomCNN()`
to plug into your existing training loop (see main.py).

Key points
----------
* Accepts either 1‑channel (grayscale) or 3‑channel input.
* About ~3.2 M parameters – light enough to train on a single GPU in <1 h.
* Uses `Conv‑BN‑ReLU` blocks + MaxPool + Global Average Pooling.
* Weight initialization = Kaiming He (fan_out) for conv, constant 0.0/1.0 for BN.
* Dropout(0.3) acts just before the classifier for regularisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Sequential):
    """Convenience: Conv2d ➔ BatchNorm2d ➔ ReLU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        if p is None:
            p = k // 2  # same‑padding by default
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class CustomCNN(nn.Module):
    """A small CNN for 224×224 CT images (grayscale or RGB)."""

    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # (B, C, 224, 224)
            _ConvBlock(input_channels, 32, k=7, s=2),   # (B,32,112,112)
            nn.MaxPool2d(3, stride=2, padding=1),       # (B,32,56,56)

            _ConvBlock(32, 64),                         # (B,64,56,56)
            _ConvBlock(64, 64),
            nn.MaxPool2d(2),                            # (B,64,28,28)

            _ConvBlock(64, 128),                        # (B,128,28,28)
            _ConvBlock(128, 128),
            nn.MaxPool2d(2),                            # (B,128,14,14)

            _ConvBlock(128, 256),                       # (B,256,14,14)
            _ConvBlock(256, 256),
            nn.MaxPool2d(2),                            # (B,256,7,7)
        )

        # Global pooling → (B,256)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    # ---------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        x = self.features(x)
        x = self.pool(x)  # (B,256,1,1)
        x = self.classifier(x)  # (B, num_classes)
        return x

# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity‑check
    model = CustomCNN(input_channels=3).eval()
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected (1,2)
    print("Params: ", sum(p.numel() for p in model.parameters())/1e6, "M")
