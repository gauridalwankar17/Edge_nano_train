"""Inference helpers for loading a model and making predictions."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import SimpleCNN


def load_labels(labels_path: str) -> Optional[List[str]]:
    """Loads class labels list from a JSON file.

    The file should have the schema: {"classes": ["airplane", ...]}
    Returns None if the file does not exist.
    """
    path = Path(labels_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
        return [str(x) for x in data["classes"]]
    return None


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 10) -> torch.nn.Module:
    """Loads the SimpleCNN model from a checkpoint path and sets it to eval mode."""
    model = SimpleCNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support either a state dict directly or a dict with key 'model_state'
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def predict_logits(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Runs a forward pass and returns (logits, inference_ms)."""
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(input_tensor.to(device))
        end = time.perf_counter()
    return logits.cpu(), (end - start) * 1000.0


def logits_to_top1(
    logits: torch.Tensor,
    labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Converts logits to top-1 prediction dictionary with confidence.

    Returns: {"index": int, "label": str, "confidence": float}
    """
    probs = F.softmax(logits.squeeze(0), dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    idx_int = int(idx.item())
    label = labels[idx_int] if labels and 0 <= idx_int < len(labels) else str(idx_int)
    return {"index": idx_int, "label": label, "confidence": float(conf.item())}

