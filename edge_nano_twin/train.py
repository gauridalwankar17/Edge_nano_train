from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .model import TinyAnomalyCNN, export_torchscript
from .preprocess import (
    WindowDataset,
    balance_classes,
    load_dataframe,
    make_windows,
    zscore_per_window,
)


def train(
    data_dir: str,
    output_model_path: str = "anomaly_cnn.pt",
    window_size: int = 256,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 8,
    patience: int = 3,
    device: str | None = None,
) -> Path:
    df, feature_cols = load_dataframe(data_dir)
    data = df[feature_cols].to_numpy()
    labels = df["label"].tolist()

    X, y = make_windows(data, labels, window_size=window_size, step=1)
    X = zscore_per_window(X)
    X, y = balance_classes(X, y)

    dataset = WindowDataset(X, y)

    # Train/Val split
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAnomalyCNN(num_features=len(feature_cols), num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Export to TorchScript
    export_path = Path(output_model_path).resolve()
    export_torchscript(model, num_features=len(feature_cols), window_size=window_size, export_path=str(export_path))
    return export_path


def main():
    parser = argparse.ArgumentParser(description="Train TinyAnomalyCNN and export TorchScript model.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing downloaded CSV files")
    parser.add_argument("--output", type=str, default="anomaly_cnn.pt", help="Path to save TorchScript model")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    export_path = train(
        data_dir=args.data_dir,
        output_model_path=args.output,
        window_size=args.window_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(f"Saved TorchScript model to {export_path}")


if __name__ == "__main__":
    main()

