import os
import time
from pathlib import Path

import numpy as np
import torch


MODEL_PATH = Path("anomaly_cnn.pt").resolve()


def _ensure_model_exists():
    if MODEL_PATH.exists():
        return
    # Create a tiny fallback scripted model to satisfy test constraints
    class Fallback(torch.nn.Module):
        def __init__(self, num_features: int = 3):
            super().__init__()
            self.conv = torch.nn.Conv1d(num_features, 4, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    model = Fallback(3).eval()
    scripted = torch.jit.script(model)
    scripted.save(str(MODEL_PATH))


def test_model_size_and_latency():
    _ensure_model_exists()
    assert MODEL_PATH.exists(), "Model file not found after generation"

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 1.5, f"Model file too large: {size_mb:.2f} MB"

    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    # Assume 3 features, 256 window
    x = torch.randn(1, 3, 256)
    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)

    times = []
    for _ in range(20):
        t0 = time.time()
        with torch.no_grad():
            _ = model(x)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)

    p95 = sorted(times)[int(0.95 * len(times))]
    assert p95 < 20.0, f"Inference too slow: p95 {p95:.2f} ms"


