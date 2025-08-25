"""CNN model definition.

Defines a simple, reasonably strong CNN for CIFAR-10–like images (3x32x32).
The architecture is intentionally compact to run on CPU while being accurate.
"""

from typing import Tuple

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A simple CNN suitable for CIFAR-10 sized inputs (3x32x32).

    Blocks:
    - Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool
    - Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool
    - Conv(64->128) -> BN -> ReLU -> MaxPool
    - Head: Flatten -> Dropout -> Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    @staticmethod
    def input_shape() -> Tuple[int, int, int]:
        """Returns expected input shape as (C, H, W)."""
        return (3, 32, 32)

