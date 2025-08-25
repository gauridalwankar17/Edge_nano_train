from typing import Tuple

import torch
from torch import nn


class TinyAnomalyCNN(nn.Module):
    """A very small 1D CNN for classifying time-series windows.

    Input shape: (batch, channels=num_features, seq_len)
    Output: logits for 3 classes
    """

    def __init__(self, num_features: int, num_classes: int = 3):
        super().__init__()
        self.num_features = num_features
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


def export_torchscript(
    model: nn.Module,
    num_features: int,
    window_size: int,
    export_path: str,
) -> None:
    """Export a model to TorchScript at export_path.

    We use scripting to avoid needing example inputs at export time.
    """
    model = model.eval().cpu()
    scripted = torch.jit.script(model)
    scripted.save(export_path)

